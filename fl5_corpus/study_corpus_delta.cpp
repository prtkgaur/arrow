// Six decode arms for PFOR-DELTA over the same 43-column corpus and the same
// page-scale working-set ladder the plain-PFOR harness uses.
//
//   seq_scal  continuous delta layout, generated scalar unpack, serial prefix sum
//   seq_simd  continuous delta layout, Arrow's shipped vectorized unpack, serial
//             prefix sum. This is the shape DELTA_BINARY_PACKED has: one
//             1024-long dependent add chain per block.
//   intlv     FastLanes interleaved container, differences taken at LANE
//             distance (v[i] - v[i-32]). 32 independent chains, output already
//             in file order, no permutation owed.
//   fl_unpk   FL_ORDER: same container, transposed lane assignment so lane l
//             holds the contiguous run [32l, 32l+32). Differences are therefore
//             ADJACENT, like the format's. Delivered in grid order -- the mode
//             the FastLanes paper operates in.
//   fl_tpos   the same bytes restored to file order, which is what a positional
//             Parquet reader must return.
//   intlv_nc  `intlv` with its scratch grid removed, as a confound control. The
//             shipped lane-delta kernel prefix-sums in a 4 KiB stack grid and
//             copies the block out; the FL_ORDER kernel writes straight to the
//             output. Comparing them as shipped would charge the layout for one
//             arm's memory round trip, so the control is what FL_ORDER has to be
//             compared against.
//
// UNLIKE the plain-PFOR harness, the arms here do NOT move the same number of
// bytes: lane-distance differences are wider than adjacent ones, so `intlv`
// pays a size cost `fl_unpk` does not. Throughput is therefore reported as
// output produced per second (values * 4 / s), which is the fair decode-speed
// metric, and bits per value is reported alongside it as the separate size
// axis. Reading the throughput columns without the bits/value columns will
// credit or blame a layout for a byte count.
//
// There is also no fl_unpk/intlv tie to check here -- that invariant only held
// for plain PFOR, where both arms packed the same residuals. What is checked
// instead is that every arm round-trips exactly, and that the two FL arms carry
// byte-identical payloads so fl_tpos/fl_unpk is the permutation alone.
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include <arrow/util/bpacking_internal.h>
#include "arrow/util/bpacking_dispatch_internal.h"
#include "arrow/util/bpacking_scalar_generated_internal.h"
#include "arrow/util/fastlanes/lane_delta.h"
#include "arrow/util/fastlanes/transposed_delta.h"

#include "corpus_generators.h"

// Everything below is private to this file. One study per translation unit,
// so a study's kernels are compiled exactly as they were when it was a
// standalone binary, and two studies can hold the same name for different
// things.
namespace {


namespace fl = arrow::util::fastlanes;
namespace bp = arrow::internal::bpacking;
using fl::LaneDeltaOrder;
using fl::TransposedBaseCoding;
using fl::TransposedRepair;
using Clock = std::chrono::steady_clock;

// The generated scalar kernel family, named as bpacking_dispatch expects.
template <typename U, int W>
using KernelScalar = arrow::internal::ScalarUnpackerForWidth<U, W>;

constexpr size_t kBlk = 1024;
constexpr int kReps = 5;
constexpr size_t kBytesPerRun = 1024u * 1024 * 1024;  // output bytes per timed run

// ---------------------------------------------------------------------------
// The continuous (Parquet-shaped) DELTA payload.
//
// Per 1024-value block: the value immediately before the block (its seed), the
// minimum delta, the width needed for (delta - min_delta), and the residuals as
// one continuous LSB-first stream. That is DELTA_BINARY_PACKED's shape with one
// block per 1024 values, and it deliberately uses the same per-block framing the
// container arms use, so the only difference between the arms is where a value's
// predecessor sits.
struct DeltaSeqPayload {
  std::vector<uint8_t> bytes;
  std::vector<uint32_t> offsets;
  std::vector<uint8_t> widths;
  std::vector<int32_t> min_delta;
  std::vector<int32_t> seed;  // the value before the block; 0 for block 0
};

static uint32_t WidthFor(uint32_t span) {
  uint32_t w = 0;
  while (w < 32 && (span >> w) != 0) ++w;
  return w;
}

static DeltaSeqPayload EncodeSequentialDelta(const std::vector<int32_t>& v) {
  const size_t nblocks = v.size() / kBlk;
  DeltaSeqPayload p;
  p.widths.resize(nblocks);
  p.min_delta.resize(nblocks);
  p.seed.resize(nblocks);
  p.offsets.resize(nblocks);

  // Deltas first, in unsigned wraparound arithmetic so no difference can
  // overflow: d[i] = v[i] - v[i-1], with v[-1] taken as 0.
  std::vector<uint32_t> d(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    const uint32_t prev = i == 0 ? 0u : static_cast<uint32_t>(v[i - 1]);
    d[i] = static_cast<uint32_t>(v[i]) - prev;
  }

  size_t off = 0;
  for (size_t b = 0; b < nblocks; ++b) {
    uint32_t mn = d[b * kBlk], mx = d[b * kBlk];
    for (size_t t = 1; t < kBlk; ++t) {
      const uint32_t x = d[b * kBlk + t];
      // Signed compare: a delta stream is signed, and framing on the signed
      // minimum is what makes the residuals small on a decreasing run.
      if (static_cast<int32_t>(x) < static_cast<int32_t>(mn)) mn = x;
      if (static_cast<int32_t>(x) > static_cast<int32_t>(mx)) mx = x;
    }
    const uint32_t w = WidthFor(mx - mn);
    p.widths[b] = static_cast<uint8_t>(w);
    p.min_delta[b] = static_cast<int32_t>(mn);
    p.seed[b] = b == 0 ? 0 : v[b * kBlk - 1];
    p.offsets[b] = static_cast<uint32_t>(off);
    off += w * kBlk / 8;
  }
  p.bytes.assign(off + 64, 0);
  for (size_t b = 0; b < nblocks; ++b) {
    const uint32_t w = p.widths[b];
    if (w == 0) continue;
    uint8_t* out = p.bytes.data() + p.offsets[b];
    uint64_t acc = 0;
    int bits = 0;
    for (size_t t = 0; t < kBlk; ++t) {
      const uint32_t r = d[b * kBlk + t] - static_cast<uint32_t>(p.min_delta[b]);
      acc |= static_cast<uint64_t>(r) << bits;
      bits += static_cast<int>(w);
      while (bits >= 8) {
        *out++ = static_cast<uint8_t>(acc);
        acc >>= 8;
        bits -= 8;
      }
    }
    if (bits) *out = static_cast<uint8_t>(acc);
  }
  return p;
}

// ---------------------------------------------------------------------------
// A scratch-free lane-delta decode, for use as a confound control.
//
// The shipped lane-delta kernel unpacks into a 4 KiB stack grid, prefix-sums
// there, and memcpys the block out -- 128 extra stores and 128 extra loads per
// block. The FL_ORDER arm's fused kernel writes straight to the output, so
// comparing the two as shipped would credit FL_ORDER with the other arm's
// memory round trip. That is the same defect already removed from the plain-PFOR
// container path. This version unpacks directly into the caller's buffer and
// prefix-sums in place, so the comparison is layout against layout.
inline void LaneDeltaDecodeNoScratch(const uint8_t* in, size_t n, int32_t* out) {
  const size_t nblocks = n / fl::kBlockSize;
  const size_t header = fl::LaneDeltaHeaderSize(n);
  const uint32_t* seeds = reinterpret_cast<const uint32_t*>(in);
  const uint8_t* widths = in + fl::kLanes * sizeof(uint32_t);
  const int32_t* mins = reinterpret_cast<const int32_t*>(
      in + fl::kLanes * sizeof(uint32_t) + ((nblocks + 3) & ~size_t(3)));
  const uint32_t* src = reinterpret_cast<const uint32_t*>(in + header);

  uint32_t carry[fl::kLanes];
  std::memcpy(carry, seeds, sizeof(carry));

  for (size_t b = 0; b < nblocks; ++b) {
    const uint32_t w = widths[b];
    const uint32_t bias = static_cast<uint32_t>(mins[b]);
    uint32_t* g = reinterpret_cast<uint32_t*>(out + b * fl::kBlockSize);

    if (w == 0) {
      for (size_t t = 0; t < fl::kBlockSize; ++t) g[t] = bias;
    } else {
      switch (w) {
#define LDNS_CASE(W)                            \
  case W:                                       \
    fl::UnpackBlock<W, true>(src, g, bias);      \
    break;
        LDNS_CASE(1) LDNS_CASE(2) LDNS_CASE(3) LDNS_CASE(4) LDNS_CASE(5)
        LDNS_CASE(6) LDNS_CASE(7) LDNS_CASE(8) LDNS_CASE(9) LDNS_CASE(10)
        LDNS_CASE(11) LDNS_CASE(12) LDNS_CASE(13) LDNS_CASE(14) LDNS_CASE(15)
        LDNS_CASE(16) LDNS_CASE(17) LDNS_CASE(18) LDNS_CASE(19) LDNS_CASE(20)
        LDNS_CASE(21) LDNS_CASE(22) LDNS_CASE(23) LDNS_CASE(24) LDNS_CASE(25)
        LDNS_CASE(26) LDNS_CASE(27) LDNS_CASE(28) LDNS_CASE(29) LDNS_CASE(30)
        LDNS_CASE(31) LDNS_CASE(32)
#undef LDNS_CASE
        default:
          break;
      }
      src += w * fl::kLanes;
    }
    for (size_t lane = 0; lane < fl::kLanes; ++lane) g[lane] += carry[lane];
    for (size_t row = 1; row < fl::kRowsPerBlock; ++row) {
      uint32_t* cur = g + row * fl::kLanes;
      const uint32_t* prev = cur - fl::kLanes;
      for (size_t lane = 0; lane < fl::kLanes; ++lane) cur[lane] += prev[lane];
    }
    std::memcpy(carry, g + (fl::kRowsPerBlock - 1) * fl::kLanes, sizeof(carry));
  }
}

// One process-wide, 4096-aligned arena. EVERY buffer this harness touches --
// all three input payloads and all output slices -- is carved out of it at a
// 4096-byte-aligned offset.
//
// The earlier version shared only the OUTPUT buffer and let each arm's input be
// its own std::vector. That left input-address-mod-4096 free to vary between
// arms and, worse, to vary with working set, because the allocator returns
// differently-placed blocks as the request grows. It produced a ratio that
// swung from 1.36x to 0.47x between two ADJACENT bit widths (11 vs 12) on
// datasets of identical shape, at the largest point and nowhere else -- an
// address artifact reported as a memory-hierarchy result.
static uint8_t* Arena(size_t bytes) {
  static uint8_t* buf = nullptr;
  static size_t cap = 0;
  if (bytes > cap) {
    free(buf);
    void* raw = nullptr;
    if (posix_memalign(&raw, 4096, bytes + 4096) != 0) abort();
    memset(raw, 0, bytes + 4096);
    buf = static_cast<uint8_t*>(raw);
    cap = bytes;
  }
  return buf;
}
static constexpr size_t RoundUpPage(size_t n) { return (n + 4095) & ~size_t(4095); }

// ---------------------------------------------------------------------------
// Arms
// ---------------------------------------------------------------------------
// Continuous layout + serial prefix sum: unpack the residuals with the frame
// add folded in (so `out` holds the deltas), then walk the block once adding
// each value to its predecessor. That second loop is the 1024-long dependent
// chain the interleaved arms cut into 32.
template <bool kScalar>
static void DecodeSeqDelta(const DeltaSeqPayload& p, const uint8_t* base, size_t n,
                           int32_t* out) {
  arrow::internal::UnpackOptions o;
  o.batch_size = static_cast<int>(kBlk);
  const size_t nblocks = n / kBlk;
  for (size_t b = 0; b < nblocks; ++b) {
    const uint32_t w = p.widths[b];
    uint32_t* dst = reinterpret_cast<uint32_t*>(out) + b * kBlk;
    const uint32_t bias = static_cast<uint32_t>(p.min_delta[b]);
    if (w == 0) {
      for (size_t t = 0; t < kBlk; ++t) dst[t] = bias;
    } else {
      o.bit_width = static_cast<int>(w);
      // The vectorized kernel over-reads by design and stops short of any byte
      // it has not been told is readable, so leaving max_read_bytes at its -1
      // default pushes the tail of EVERY vector onto a one-value-at-a-time
      // scalar epilog. 128*w -- what -1 deduces -- would change nothing; the
      // bound has to name bytes PAST the block. A real reader can, because the
      // packed stream continues to the end of the page.
      const size_t readable = p.bytes.size() - p.offsets[b];
      o.max_read_bytes = static_cast<int>(std::min<size_t>(
          readable, static_cast<size_t>(std::numeric_limits<int>::max())));
      if (kScalar) {
        bp::unpack_jump<KernelScalar, /*kHasBias=*/true>(base + p.offsets[b], dst, o,
                                                         bias);
      } else {
        arrow::internal::unpack_bias<uint32_t>(base + p.offsets[b], dst, o, bias);
      }
    }
    uint32_t running = static_cast<uint32_t>(p.seed[b]);
    for (size_t t = 0; t < kBlk; ++t) {
      running += dst[t];
      dst[t] = running;
    }
  }
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------
struct Row {
  std::string dataset;
  const char* point;
  size_t n;
  double gibs[6];
  double bpv[3];   // bits per value: continuous, lane-distance, FL_ORDER
  double avg_w;
  double src_mib;  // distinct packed bytes streamed
  double dst_mib;  // distinct output bytes written
};

static const char* kArm[6] = {"seq_scal", "seq_simd", "intlv",
                              "fl_unpk",  "fl_tpos",  "intlv_nc"};

struct Dataset {
  const char* name;
  std::vector<int32_t> (*gen)(int64_t);
};

#define D(Name) {#Name, corpus::Gen##Name}
#define DS(Name) {#Name, corpus::delta_shapes::Gen##Name<int32_t>}
static const Dataset kDatasets[] = {
    // ClickBench-inspired
    D(ClientIP), D(UrlRegionID), D(CounterID), D(EventDate), D(EventTime),
    D(GoodEvent), D(HID), D(HitColor), D(IPNetworkID), D(JavaEnable), D(OS),
    D(Resolution), D(TrafficSourceID), D(UserAgent),
    // TPC-DS
    D(TpcdsSoldDateSk), D(TpcdsStoreSk), D(TpcdsItemSk), D(TpcdsQuantity),
    D(TpcdsCustomerSk), D(TpcdsExtSalesPrice), D(TpcdsNetProfit), D(TpcdsDYear),
    // TPC-H
    D(TpchLQuantity), D(TpchLExtendedPrice), D(TpchLDiscount), D(TpchLShipDate),
    // NYC taxi
    D(TaxiPickupUnixTime), D(TaxiTripDistanceX100), D(TaxiFareCents),
    // correlated / sorted
    D(SortedUnixTime), D(SortedKeyDups), D(MonotoneRowId), D(NearSortedUnixTime),
    // shapes with structure between neighbouring values
    DS(TrendJitter), DS(Sawtooth), {"MeasurementSeries", corpus::delta_shapes::GenMeasurement<int32_t>},
    DS(SortedKeys), DS(SensorDropouts), DS(IdsWithGaps), DS(RandomWalk),
    DS(EventMillis), DS(LowSentinel), DS(Bimodal),
};
#undef D
#undef DS
static constexpr size_t kNumDatasets = sizeof(kDatasets) / sizeof(kDatasets[0]);

}  // namespace


int RunCorpusDeltaStudy(int argc, char** argv) {
  // The ladder is built around the unit a Parquet reader actually decodes: one
  // data page, which defaults to 1 MiB of ENCODED bytes and is therefore always
  // small. A 32-MiB decode call does not occur in any reader, so growing `n` to
  // 32 MiB does not model "out of cache" -- it models nothing.
  //
  // What genuinely does leave cache is the STREAM: a scan walks page after page
  // and never revisits one, so every page's input is cold however small the page
  // is. So this ladder holds the decode call at page scale and grows the stream
  // instead, by rotating over `src_copies` distinct copies of the payload. It
  // also names both footprints on every line, because labelling a row by its
  // output size while the input moves too is how the previous ladder produced a
  // number nobody could attribute.
  struct Point {
    const char* name;
    size_t n;           // values per decode call -- the page-scale decode unit
    size_t src_target;  // bytes of DISTINCT packed input to rotate over (0 = one copy)
    size_t dst_target;  // bytes of DISTINCT output to rotate over (0 = one copy)
  };
  const Point kPoints[] = {
      {"page16k", 4 * kBlk, 0, 0},
      {"page256k", 64 * kBlk, 0, 0},
      {"page1m", 256 * kBlk, 0, 0},               // ~1 MiB out: a full default page
      {"scan4m", 256 * kBlk, 4ull << 20, 0},      // same page, 4 MiB of cold source
      {"scan48m", 256 * kBlk, 48ull << 20, 0},    // same page, source past this L3
      {"batch48m", 256 * kBlk, 0, 48ull << 20},   // same page, destination past L3
  };

  const char* only = (argc > 1) ? argv[1] : nullptr;
  const char* csv_path = (argc > 2) ? argv[2] : nullptr;
  FILE* csv = csv_path ? fopen(csv_path, "w") : nullptr;
  if (csv)
    fprintf(csv,
            "dataset,point,n,avg_bit_width,src_mib,dst_mib,seq_scal,seq_simd,intlv,"
            "fl_unpk,fl_tpos,intlv_nc,bpv_seq,bpv_intlv,bpv_fl\n");

  printf("# delta -- delta everywhere, six decoders, best-of-%d\n", kReps);
  printf("# one shared 4096-aligned output buffer; throughput is OUTPUT bytes per\n"
         "# second (n*4), because the arms do not move the same input bytes.\n");
  printf("# bpv columns carry the size axis: lane-distance differences are wider\n"
         "# than adjacent ones, so the fastest arm is not automatically the best.\n");
#ifdef ARROW_TRANSPOSED_DELTA_FUSED_REPAIR
  printf("# fl_tpos: FUSED repair -- prefix sum and transpose in one pass.\n");
#else
  printf("# fl_tpos: SEPARATE repair -- prefix sum, then a second pass to\n"
         "#          transpose. No fused repair on this target, so this arm is a\n"
         "#          LOWER BOUND on FL_ORDER and not its real cost.\n");
#endif
  printf("# %zu datasets x %zu working sets\n#\n", kNumDatasets,
         sizeof(kPoints) / sizeof(kPoints[0]));

  const char* hdr_fmt =
      "%-22s %-9s %5s %8s %8s | %7s %7s %7s %7s %7s %8s | %7s %7s %7s | %6s %6s %6s\n";
  printf(hdr_fmt, "dataset", "point", "W", "srcMiB", "dstMiB", kArm[0], kArm[1], kArm[2],
         kArm[3], kArm[4], kArm[5], "int/sd", "unpk/sd", "tpos/sd", "bpvSeq", "bpvInt",
         "bpvFL");
  std::string dashes(170, '-');
  printf("%s\n", dashes.c_str());

  std::vector<Row> rows;
  for (size_t d = 0; d < kNumDatasets; ++d) {
    const Dataset& ds = kDatasets[d];
    if (only && *only && strcmp(only, "all") != 0 && strstr(ds.name, only) == nullptr)
      continue;

    for (const Point& pt : kPoints) {
      const size_t n = pt.n;
      const std::vector<int32_t> values = ds.gen(static_cast<int64_t>(n));
      if (values.size() != n) {
        fprintf(stderr, "generator %s returned %zu for n=%zu\n", ds.name,
                values.size(), n);
        return 1;
      }

      const DeltaSeqPayload seq = EncodeSequentialDelta(values);
      std::vector<uint8_t> ld(fl::LaneDeltaMaxEncodedSize(n));
      std::vector<uint8_t> td(fl::TransposedMaxEncodedSize(n));
      const size_t ld_len =
          fl::LaneDeltaEncode<LaneDeltaOrder::kInterleaved>(values.data(), n, ld.data());
      const size_t td_len = fl::TransposedDeltaEncode<TransposedBaseCoding::kPacked>(
          values.data(), n, td.data());
      if (ld_len == 0 || td_len == 0) {
        fprintf(stderr, "encode failed for %s (lane=%zu tposed=%zu)\n", ds.name, ld_len,
                td_len);
        return 1;
      }

      double avg_w = 0;
      for (uint8_t w : seq.widths) avg_w += w;
      avg_w /= static_cast<double>(seq.widths.size());
      // Bits per value INCLUDING each scheme's own side data, which is where the
      // schemes really differ: the continuous arm carries one min-delta and one
      // seed per block, the lane arm 32 seeds per column plus a min per block,
      // the FL_ORDER arm 32 bases per block (delta-coded across blocks).
      const double bpv_seq =
          static_cast<double>(seq.bytes.size() - 64 +
                              seq.widths.size() * (sizeof(int32_t) * 2 + 1)) *
          8 / static_cast<double>(n);
      const double bpv_ld = static_cast<double>(ld_len) * 8 / static_cast<double>(n);
      const double bpv_td = static_cast<double>(td_len) * 8 / static_cast<double>(n);

      // --- one arena, every buffer page-aligned inside it ---------------------
      // All three payload strides are page-rounded, so copy k of each arm's
      // input sits at the same offset mod 4096. Input placement is therefore
      // held fixed across arms and across working sets instead of being left to
      // the allocator.
      // Strides are page-rounded and then SKEWED by one page. Without the skew a
      // 1 MiB output slice stride is an exact power of two, so every slice maps
      // onto the same cache sets and the largest point measures set conflicts
      // rather than bandwidth -- it read 0.36x on one column and 1.55x on
      // another of the same width. One page of skew walks the set index forward
      // per slice and costs 0.4% of footprint.
      const size_t kSkew = 4096;
      const size_t seq_stride = RoundUpPage(seq.bytes.size()) + kSkew;
      const size_t ld_stride = RoundUpPage(ld_len) + kSkew;
      const size_t td_stride = RoundUpPage(td_len) + kSkew;
      const size_t out_stride = RoundUpPage(n * sizeof(int32_t)) + kSkew;
      auto copies_for = [](size_t target, size_t unit) {
        return (target == 0 || unit == 0) ? size_t{1}
                                          : std::max<size_t>(1, (target + unit - 1) / unit);
      };
      // The source footprint is set by the largest of the three payloads so all
      // three arms stream the same number of distinct bytes.
      const size_t src_unit = std::max(seq_stride, std::max(ld_stride, td_stride));
      const size_t src_copies = copies_for(pt.src_target, src_unit);
      const size_t dst_copies = copies_for(pt.dst_target, out_stride);

      uint8_t* arena = Arena(src_copies * (seq_stride + ld_stride + td_stride) +
                             dst_copies * out_stride);
      // The output goes first, so out_base is the same address for every dataset
      // and every point. Placing it after the payloads made its offset a
      // function of the payload sizes, which is a per-dataset variable in a
      // measurement that is supposed to hold placement fixed.
      int32_t* out_base = reinterpret_cast<int32_t*>(arena);
      uint8_t* seq_base = arena + dst_copies * out_stride;
      uint8_t* ld_base = seq_base + src_copies * seq_stride;
      uint8_t* td_base = ld_base + src_copies * ld_stride;
      for (size_t k = 0; k < src_copies; ++k) {
        memcpy(seq_base + k * seq_stride, seq.bytes.data(), seq.bytes.size());
        memcpy(ld_base + k * ld_stride, ld.data(), ld_len);
        memcpy(td_base + k * td_stride, td.data(), td_len);
      }
      const double src_mib =
          static_cast<double>(src_copies * src_unit) / (1024.0 * 1024);
      const double dst_mib =
          static_cast<double>(dst_copies * out_stride) / (1024.0 * 1024);

      // Each arm decodes copy (it % src_copies) into slice (it % dst_copies), so
      // a "scan" point really does touch distinct cold bytes every iteration.
      auto slot_out = [&](size_t it) {
        return reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(out_base) +
                                         (it % dst_copies) * out_stride);
      };
      auto a_seq_scal = [&](size_t it) {
        DecodeSeqDelta<true>(seq, seq_base + (it % src_copies) * seq_stride, n,
                             slot_out(it));
      };
      auto a_seq_simd = [&](size_t it) {
        DecodeSeqDelta<false>(seq, seq_base + (it % src_copies) * seq_stride, n,
                              slot_out(it));
      };
      auto a_intlv = [&](size_t it) {
        fl::LaneDeltaDecode<LaneDeltaOrder::kInterleaved>(
            ld_base + (it % src_copies) * ld_stride, n, slot_out(it));
      };
      auto a_fl_unpk = [&](size_t it) {
        fl::TransposedDeltaDecode<TransposedBaseCoding::kPacked, TransposedRepair::kNone>(
            td_base + (it % src_copies) * td_stride, n, slot_out(it));
      };
      auto a_fl_tpos = [&](size_t it) {
        fl::TransposedDeltaDecode<TransposedBaseCoding::kPacked,
                                  TransposedRepair::kFused>(
            td_base + (it % src_copies) * td_stride, n, slot_out(it));
      };
      auto a_intlv_nc = [&](size_t it) {
        LaneDeltaDecodeNoScratch(ld_base + (it % src_copies) * ld_stride, n,
                                 slot_out(it));
      };
      std::function<void(size_t)> arms[6] = {a_seq_scal, a_seq_simd, a_intlv,
                                             a_fl_unpk,  a_fl_tpos,  a_intlv_nc};
      int32_t* out = out_base;

      // --- correctness before speed -----------------------------------------
      // Five of the six arms must reproduce `values` exactly. fl_unpk returns FL
      // order on purpose, so it is checked against the FL_ORDER permutation of
      // `values` instead of against `values`. These are round-trip checks: each
      // arm decodes what its own encoder wrote, so a wrong width choice, a
      // dropped carry or a mis-seeded block shows up here rather than as speed.
      for (int a = 0; a < 6; ++a) {
        memset(out, 0xCD, n * sizeof(int32_t));
        arms[a](0);
        if (a == 3) continue;  // fl_unpk: order-agnostic, checked below
        if (memcmp(out, values.data(), n * sizeof(int32_t)) != 0) {
          size_t bad = 0;
          while (bad < n && out[bad] == values[bad]) ++bad;
          fprintf(stderr, "MISMATCH %s/%s arm=%s at %zu: got %d want %d\n", ds.name,
                  pt.name, kArm[a], bad, out[bad], values[bad]);
          return 1;
        }
      }
      {
        // fl_unpk must equal the FL_ORDER permutation of the input: for each
        // block, lane l holds the contiguous run [32l, 32l+32).
        memset(out, 0xCD, n * sizeof(int32_t));
        a_fl_unpk(0);
        for (size_t b = 0; b < n / kBlk; ++b) {
          for (size_t lane = 0; lane < 32; ++lane) {
            for (size_t r = 0; r < 32; ++r) {
              const int32_t got = out[b * kBlk + r * 32 + lane];
              const int32_t want = values[b * kBlk + lane * 32 + r];
              if (got != want) {
                fprintf(stderr, "MISMATCH %s/%s arm=fl_unpk blk=%zu lane=%zu row=%zu: "
                                "got %d want %d\n", ds.name, pt.name, b, lane, r, got,
                        want);
                return 1;
              }
            }
          }
        }
      }

      // --- timing -----------------------------------------------------------
      // Every timed run moves ~1 GiB of output regardless of working set, so the
      // shortest run (L1, fastest arm) is still ~10 ms. The six arms alternate
      // within each repetition, so drift or thermal effects hit all of them alike.
      // Enough iterations to move ~1 GiB and to visit every distinct copy.
      const size_t iters = std::max(std::max(src_copies, dst_copies),
                                    std::max<size_t>(3, kBytesPerRun / (n * 4)));
      double best[6] = {0, 0, 0, 0, 0, 0};
      for (int rep = 0; rep < kReps; ++rep) {
        for (int a = 0; a < 6; ++a) {
          for (size_t k = 0; k < std::max(src_copies, dst_copies); ++k) arms[a](k);  // warm
          const auto t0 = Clock::now();
          for (size_t it = 0; it < iters; ++it) arms[a](it);
          const auto t1 = Clock::now();
          const double secs = std::chrono::duration<double>(t1 - t0).count();
          const double gibs =
              static_cast<double>(iters) * n * 4 / secs / (1024.0 * 1024 * 1024);
          best[a] = std::max(best[a], gibs);
        }
      }

      Row row;
      row.dataset = ds.name;
      row.point = pt.name;
      row.n = n;
      for (int a = 0; a < 6; ++a) row.gibs[a] = best[a];
      row.avg_w = avg_w;
      row.src_mib = src_mib;
      row.dst_mib = dst_mib;
      row.bpv[0] = bpv_seq;
      row.bpv[1] = bpv_ld;
      row.bpv[2] = bpv_td;
      rows.push_back(row);

      printf("%-22s %-9s %5.1f %8.2f %8.2f | %7.1f %7.1f %7.1f %7.1f %7.1f %8.1f |"
             " %6.2fx %6.2fx %6.2fx | %6.2f %6.2f %6.2f\n",
             ds.name, pt.name, avg_w, src_mib, dst_mib, best[0], best[1], best[2],
             best[3], best[4], best[5], best[2] / best[1], best[3] / best[1],
             best[4] / best[1], bpv_seq, bpv_ld, bpv_td);
      fflush(stdout);
      if (csv) {
        fprintf(csv,
                "%s,%s,%zu,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.4f,%.4f,"
                "%.4f\n",
                ds.name, pt.name, n, avg_w, src_mib, dst_mib, best[0], best[1], best[2],
                best[3], best[4], best[5], bpv_seq, bpv_ld, bpv_td);
        fflush(csv);
      }
    }
  }

  // --- geomeans per working set --------------------------------------------
  printf("%s\n", dashes.c_str());
  for (const Point& pt : kPoints) {
    double g[6] = {1, 1, 1, 1, 1, 1};
    double gb[3] = {1, 1, 1};
    int cnt = 0;
    for (const Row& r : rows) {
      if (strcmp(r.point, pt.name) != 0) continue;
      for (int a = 0; a < 6; ++a) g[a] *= r.gibs[a];
      for (int k = 0; k < 3; ++k) gb[k] *= r.bpv[k];
      ++cnt;
    }
    if (cnt == 0) continue;
    double m[6], mb[3];
    for (int a = 0; a < 6; ++a) m[a] = std::pow(g[a], 1.0 / cnt);
    for (int k = 0; k < 3; ++k) mb[k] = std::pow(gb[k], 1.0 / cnt);
    printf("%-22s %-9s %5s %8s %8s | %7.1f %7.1f %7.1f %7.1f %7.1f %8.1f |"
           " %6.2fx %6.2fx %6.2fx | %6.2f %6.2f %6.2f   (n=%d)\n",
           "GEOMEAN", pt.name, "", "", "", m[0], m[1], m[2], m[3], m[4], m[5],
           m[2] / m[1], m[3] / m[1], m[4] / m[1], mb[0], mb[1], mb[2], cnt);
  }

  // --- what the two control arms price -------------------------------------
  //
  // The plain-PFOR harness could self-check its buffer placement, because there
  // the FL_ORDER-raw arm and the interleaved arm run the identical kernel over
  // the identical byte count and have to tie: any point where they did not was
  // measuring placement. Delta has no such pair -- every arm here differs from
  // every other in either the byte count or the kernel -- so that check does not
  // exist and is not faked. What stands in for it is that the arena, the page
  // rounding and the one-page skew are the same code as the harness that does
  // pass its tie, and that both control arms below are differences of one thing.
  //
  //   scratch  intlv_nc / intlv. Same layout, same bytes, same prefix sum; the
  //            only difference is that the shipped kernel prefix-sums in a 4 KiB
  //            stack grid and memcpys the block out, and the control writes
  //            straight to the output. This is the round trip, alone. Any
  //            comparison against the FL_ORDER arms has to use the control,
  //            because the FL_ORDER kernel already writes straight out.
  //
  //   permute  fl_tpos / fl_unpk. Byte-identical payload, same unpack, same
  //            prefix sum; the only difference is that one restores file order
  //            and the other does not. This is what a positional reader pays for
  //            FL_ORDER, and it is the number the paper never has to quote,
  //            because it never restores order.
  printf("\ncontrols (both are one-variable differences):\n");
  printf("  %-9s %8s %8s   %8s %8s\n", "point", "scratch", "worst", "permute", "worst");
  for (const Point& pt : kPoints) {
    double ls = 0, lp = 0;
    double ws = 1, wp = 1;
    std::string wsd, wpd;
    int cnt = 0;
    for (const Row& r : rows) {
      if (strcmp(r.point, pt.name) != 0) continue;
      const double qs = r.gibs[5] / r.gibs[2];
      const double qp = r.gibs[4] / r.gibs[3];
      ls += std::log(qs);
      lp += std::log(qp);
      ++cnt;
      if (std::fabs(std::log(qs)) > std::fabs(std::log(ws))) {
        ws = qs;
        wsd = r.dataset;
      }
      if (std::fabs(std::log(qp)) > std::fabs(std::log(wp))) {
        wp = qp;
        wpd = r.dataset;
      }
    }
    if (cnt == 0) continue;
    printf("  %-9s %7.3fx %7.3fx %s   %7.3fx %7.3fx %s   (n=%d)\n", pt.name,
           std::exp(ls / cnt), ws, wsd.c_str(), std::exp(lp / cnt), wp, wpd.c_str(),
           cnt);
  }

  if (csv) fclose(csv);
  return 0;
}
