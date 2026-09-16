// Synthetic FOR/bit-unpacking microbenchmark: no exceptions, delta, page
// decompression, or downstream consumer. Ratios include dispatch and framing.
// Five decoders plus a write-only reference, not an end-to-end Parquet result.
#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include <arrow/util/bpacking_internal.h>
#include "arrow/util/bpacking_dispatch_internal.h"
#include "arrow/util/bpacking_scalar_generated_internal.h"
#include "arrow/util/fastlanes/interleaved_pfor.h"

#include "corpus_generators.h"
#include "timing.h"

namespace fl = arrow::util::fastlanes;
namespace bp = arrow::internal::bpacking;
using fl::InterleavedPforOrder;
using Clock = std::chrono::steady_clock;

// The generated scalar kernel family, named as bpacking_dispatch expects.
template <typename U, int W>
using KernelScalar = arrow::internal::ScalarUnpackerForWidth<U, W>;

constexpr size_t kBlk = 1024;

// ---------------------------------------------------------------------------
// The sequential (Parquet-shaped) payload.
//
// Per 1024-value block: the block minimum, the bit width needed for
// (value - min), and the residuals as one continuous LSB-first stream. That is
// the frame-of-reference PFOR layout Parquet's bit-packed pages use, and it is
// deliberately the SAME per-block min and SAME per-block width the FastLanes
// container picks, so the sequential and container arms decode the same values
// at the same widths and the only difference is the physical layout.
// ---------------------------------------------------------------------------
struct SeqPayload {
  std::vector<uint8_t> bytes;     // packed residuals, block-aligned
  std::vector<uint32_t> offsets;  // byte offset of each block's residuals
  std::vector<uint8_t> widths;
  std::vector<int32_t> mins;
};

static uint32_t WidthFor(uint32_t span) {
  uint32_t w = 0;
  while (w < 32 && (span >> w) != 0) ++w;
  return w;
}

static SeqPayload EncodeSequential(const std::vector<int32_t>& v) {
  const size_t nblocks = v.size() / kBlk;
  SeqPayload p;
  p.widths.resize(nblocks);
  p.mins.resize(nblocks);
  p.offsets.resize(nblocks);
  // Blocks are byte-aligned so each block can be handed to unpack_bias
  // independently -- which is also how a real page decoder walks them.
  size_t off = 0;
  for (size_t b = 0; b < nblocks; ++b) {
    int32_t mn = v[b * kBlk], mx = v[b * kBlk];
    for (size_t t = 1; t < kBlk; ++t) {
      mn = std::min(mn, v[b * kBlk + t]);
      mx = std::max(mx, v[b * kBlk + t]);
    }
    const uint32_t span = static_cast<uint32_t>(mx) - static_cast<uint32_t>(mn);
    const uint32_t w = WidthFor(span);
    p.widths[b] = static_cast<uint8_t>(w);
    p.mins[b] = mn;
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
      const uint32_t r =
          static_cast<uint32_t>(v[b * kBlk + t]) - static_cast<uint32_t>(p.mins[b]);
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

// Reuse one aligned arena. Reallocation can move it between cases; within a
// case every arm uses the identical source and destination addresses. Payloads
// are copied into the source outside timing, then warmed identically.
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
template <bool kScalar>
static void DecodeSeq(const SeqPayload& p, const uint8_t* base, size_t n, int32_t* out) {
  arrow::internal::UnpackOptions o;
  o.batch_size = static_cast<int>(kBlk);
  const size_t nblocks = n / kBlk;
  for (size_t b = 0; b < nblocks; ++b) {
    const uint32_t w = p.widths[b];
    uint32_t* dst = reinterpret_cast<uint32_t*>(out) + b * kBlk;
    const uint32_t bias = static_cast<uint32_t>(p.mins[b]);
    if (w == 0) {
      for (size_t t = 0; t < kBlk; ++t) dst[t] = bias;
      continue;
    }
    o.bit_width = static_cast<int>(w);
    // The vectorized kernel over-reads by design and stops short of any byte it
    // has not been told is readable, so leaving max_read_bytes at its -1 default
    // pushes the tail of EVERY vector onto a one-value-at-a-time scalar epilog.
    // A block needs exactly 128*w bytes, which is what -1 deduces, so quoting
    // 128*w here would change nothing: the bound has to name bytes PAST the
    // block for the last vector step to be allowed. A real reader can, and does
    // -- the packed stream continues to the end of the page -- so what is passed
    // is the distance from this block to the end of the payload, the same
    // quantity the production decoder derives from its page span.
    const size_t readable = p.bytes.size() - p.offsets[b];
    o.max_read_bytes = static_cast<int>(
        std::min<size_t>(readable, static_cast<size_t>(std::numeric_limits<int>::max())));
    if (kScalar) {
      bp::unpack_jump<KernelScalar, /*kHasBias=*/true>(base + p.offsets[b], dst, o, bias);
    } else {
      arrow::internal::unpack_bias<uint32_t>(base + p.offsets[b], dst, o, bias);
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
  double cr;  // 32 / mean bits per value, from the container encoding
  double avg_w;
  double src_mib;  // maximum per-arm payload bytes (excludes untouched padding)
  double dst_mib;  // distinct output bytes written
};

static constexpr int kArms = 6;
static const char* kArm[kArms] = {"seq_scal", "seq_simd", "intlv",
                                  "fl_unpk",  "fl_tpos",  "pure_st"};

struct Dataset {
  const char* name;
  std::vector<int32_t> (*gen)(int64_t);
};

#define D(Name) {#Name, corpus::Gen##Name}
#define DS(Name) {#Name, corpus::delta_shapes::Gen##Name<int32_t>}
static const Dataset kDatasets[] = {
    // ClickBench-inspired
    D(ClientIP),
    D(UrlRegionID),
    D(CounterID),
    D(EventDate),
    D(EventTime),
    D(GoodEvent),
    D(HID),
    D(HitColor),
    D(IPNetworkID),
    D(JavaEnable),
    D(OS),
    D(Resolution),
    D(TrafficSourceID),
    D(UserAgent),
    // TPC-DS
    D(TpcdsSoldDateSk),
    D(TpcdsStoreSk),
    D(TpcdsItemSk),
    D(TpcdsQuantity),
    D(TpcdsCustomerSk),
    D(TpcdsExtSalesPrice),
    D(TpcdsNetProfit),
    D(TpcdsDYear),
    // TPC-H
    D(TpchLQuantity),
    D(TpchLExtendedPrice),
    D(TpchLDiscount),
    D(TpchLShipDate),
    // NYC taxi
    D(TaxiPickupUnixTime),
    D(TaxiTripDistanceX100),
    D(TaxiFareCents),
    // correlated / sorted
    D(SortedUnixTime),
    D(SortedKeyDups),
    D(MonotoneRowId),
    D(NearSortedUnixTime),
    // shapes with structure between neighbouring values
    DS(TrendJitter),
    DS(Sawtooth),
    {"MeasurementSeries", corpus::delta_shapes::GenMeasurement<int32_t>},
    DS(SortedKeys),
    DS(SensorDropouts),
    DS(IdsWithGaps),
    DS(RandomWalk),
    DS(EventMillis),
    DS(LowSentinel),
    DS(Bimodal),
};
#undef D
#undef DS
static constexpr size_t kNumDatasets = sizeof(kDatasets) / sizeof(kDatasets[0]);

int main(int argc, char** argv) {
  const auto options = bench::Parse(argc, argv);
  auto rng = std::mt19937(options.seed);
  // These are DECODED sizes, not encoded Parquet page sizes. Rotating source
  // and destination footprints separate the two effects. Residency depends on
  // the local caches; 48 MiB is not universally beyond LLC. This is single-threaded.
  struct Point {
    const char* name;
    size_t n;           // values per decode call -- the page-scale decode unit
    size_t src_target;  // bytes of DISTINCT packed input to rotate over (0 = one copy)
    size_t dst_target;  // bytes of DISTINCT output to rotate over (0 = one copy)
  };
  const Point kPoints[] = {
      {"page16k", 4 * kBlk, 0, 0},
      {"page256k", 64 * kBlk, 0, 0},
      {"page1m", 256 * kBlk, 0, 0},              // 1 MiB decoded output
      {"scan4m", 256 * kBlk, 4ull << 20, 0},     // rotate 4 MiB of source payload
      {"scan48m", 256 * kBlk, 48ull << 20, 0},   // rotate 48 MiB of source payload
      {"batch48m", 256 * kBlk, 0, 48ull << 20},  // rotate 48 MiB of decoded output
  };

  const char* only = options.filter.c_str();
  FILE* csv = bench::Open(options.csv);
  FILE* raw = bench::Open(options.csv.empty() ? "" : options.csv + ".raw.csv");
  if (csv)
    fprintf(csv,
            "dataset,point,n,avg_bit_width,cr,src_mib,dst_mib,seq_scal,seq_simd,intlv,fl_"
            "unpk,fl_tpos,pure_st,max_cv,control_ratio,valid,repetitions,seed,seq_src_"
            "mib,grid_src_mib,src_span_mib,dst_span_mib\n");
  if (raw) fprintf(raw, "dataset,point,arm,repetition,order,iterations,seconds,gibs\n");
  size_t selected = 0;
  printf("# fl5_corpus -- bit-unpacking only, no delta anywhere\n");
  printf(
      "# five decode arms plus a write-only reference, median-of-%d; seed=%u; "
      "verify_only=%d\n",
      options.reps, options.seed, options.verify);
  printf(
      "# pure_st writes the same bytes to the same place with no unpacking,\n"
      "# it omits source reads and is a reference, not a measured bandwidth ceiling.\n");
#ifdef ARROW_FASTLANES_FUSED_FL_UNPACK
  printf("# fl_tpos: FUSED in-register transpose (UnpackBlockFlToFileOrder)\n");
#else
  printf(
      "# fl_tpos: UNFUSED fallback -- UnpackBlock into a 4 KiB scratch grid,\n"
      "#          then Transpose32x32 out of it. No fused kernel on this target,\n"
      "#          this is the actual implementation measured on this target.\n");
#endif
  printf("# %zu datasets x %zu working sets\n#\n", kNumDatasets,
         sizeof(kPoints) / sizeof(kPoints[0]));

  const char* hdr_fmt =
      "%-22s %-9s %5s %5s %8s %8s | %8s %8s %8s %8s %8s %8s |"
      " %8s %8s %8s %8s %8s %8s\n";
  printf(hdr_fmt, "dataset", "point", "W", "cr", "srcMiB", "dstMiB", kArm[0], kArm[1],
         kArm[2], kArm[3], kArm[4], kArm[5], "unpk/sc", "unpk/sd", "unpk/int", "tpos/sd",
         "sd/store", "int/store");
  std::string dashes(184, '-');
  printf("%s\n", dashes.c_str());

  std::vector<Row> rows;
  bool noisy = false;
  for (size_t d = 0; d < kNumDatasets; ++d) {
    const Dataset& ds = kDatasets[d];
    if (strcmp(only, "all") != 0 &&
        (options.exact ? strcmp(ds.name, only) != 0 : strstr(ds.name, only) == nullptr))
      continue;

    ++selected;
    for (const Point& pt : kPoints) {
      const size_t n = pt.n;
      const std::vector<int32_t> values = ds.gen(static_cast<int64_t>(n));
      if (values.size() != n) {
        fprintf(stderr, "generator %s returned %zu for n=%zu\n", ds.name, values.size(),
                n);
        return 1;
      }

      const SeqPayload seq = EncodeSequential(values);
      std::vector<uint8_t> ib(fl::InterleavedPforMaxEncodedSize(n));
      std::vector<uint8_t> fb(fl::InterleavedPforMaxEncodedSize(n));
      const size_t ib_len = fl::InterleavedPforEncode<InterleavedPforOrder::kFileOrder>(
          values.data(), n, ib.data());
      const size_t fb_len = fl::InterleavedPforEncode<InterleavedPforOrder::kFlOrder>(
          values.data(), n, fb.data());
      if (ib_len == 0 || fb_len == 0) {
        fprintf(stderr, "encode failed for %s\n", ds.name);
        return 1;
      }
      if (ib_len != fb_len) {
        fprintf(stderr,
                "ERROR %s: interleaved and FL_ORDER wire sizes differ "
                "(%zu vs %zu) -- these arms are not decoding the same "
                "number of bytes\n",
                ds.name, ib_len, fb_len);
        return 1;
      }

      double avg_w = 0;
      for (uint8_t w : seq.widths) avg_w += w;
      avg_w /= static_cast<double>(seq.widths.size());

      // Identical source address and stride for all arms; refill outside timing.
      // Page skew avoids a power-of-two stride, but does not guarantee identical
      // cache behaviour on every CPU. The control check remains necessary.
      const size_t payload = std::max(seq.bytes.size(), ib_len);
      const size_t src_stride = RoundUpPage(payload) + 4096;
      const size_t out_stride = RoundUpPage(n * sizeof(int32_t)) + 4096;
      auto copies_for = [](size_t target, size_t bytes) {
        return target == 0 ? size_t{1}
                           : std::max<size_t>(1, (target + bytes - 1) / bytes);
      };
      // Size targets count payload bytes actually read/written, not stride gaps.
      const size_t src_copies =
          copies_for(pt.src_target, std::min(seq.bytes.size(), ib_len));
      const size_t dst_copies = copies_for(pt.dst_target, n * sizeof(int32_t));
      uint8_t* arena = Arena(src_copies * src_stride + dst_copies * out_stride);
      int32_t* out_base = reinterpret_cast<int32_t*>(arena);
      uint8_t* src_base = arena + dst_copies * out_stride;
      auto prepare = [&](int arm) {
        const uint8_t* data = arm < 2    ? seq.bytes.data()
                              : arm == 2 ? ib.data()
                                         : fb.data();
        const size_t bytes = arm < 2 ? seq.bytes.size() : arm == 2 ? ib_len : fb_len;
        for (size_t k = 0; k < src_copies; ++k)
          memcpy(src_base + k * src_stride, data, bytes);
      };
      const double src_mib = static_cast<double>(src_copies * payload) / (1 << 20);
      const double dst_mib = static_cast<double>(dst_copies * n * 4) / (1 << 20);

      auto slot_out = [&](size_t it) {
        return reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(out_base) +
                                          (it % dst_copies) * out_stride);
      };
      auto a_seq_scal = [&](size_t it) {
        DecodeSeq<true>(seq, src_base + (it % src_copies) * src_stride, n, slot_out(it));
      };
      auto a_seq_simd = [&](size_t it) {
        DecodeSeq<false>(seq, src_base + (it % src_copies) * src_stride, n, slot_out(it));
      };
      auto a_intlv = [&](size_t it) {
        fl::InterleavedPforDecode<InterleavedPforOrder::kFileOrder>(
            src_base + (it % src_copies) * src_stride, n, slot_out(it));
      };
      // Raw FL order needs exactly the same decoder, not a separately compiled
      // template/lambda whose code placement could bias the identity control.
      auto a_fl_unpk = a_intlv;
      auto a_fl_tpos = [&](size_t it) {
        fl::InterleavedPforDecode<InterleavedPforOrder::kFlOrder>(
            src_base + (it % src_copies) * src_stride, n, slot_out(it));
      };
      // Write-only reference: same destination, but no packed-input stream.
      // This still reads the small mins array and performs vector arithmetic.
      auto a_pure_st = [&](size_t it) {
        uint32_t* dst = reinterpret_cast<uint32_t*>(slot_out(it));
        for (size_t b = 0; b < n / kBlk; ++b) {
          const uint32_t bias = static_cast<uint32_t>(seq.mins[b]);
          uint32_t* q = dst + b * kBlk;
          for (size_t t = 0; t < kBlk; ++t) q[t] = bias + static_cast<uint32_t>(t);
        }
      };
      std::function<void(size_t)> arms[kArms] = {a_seq_scal, a_seq_simd, a_intlv,
                                                 a_fl_unpk,  a_fl_tpos,  a_pure_st};
      // Validate every source copy and every destination slice before timing,
      // including the permuted arm and the write-only reference. A canary checks
      // that no arm writes past its output slice.
      const size_t visits = std::max(src_copies, dst_copies);
      for (int a = 0; a < kArms; ++a) {
        prepare(a);
        for (size_t k = 0; k < visits; ++k) {
          int32_t* out = slot_out(k);
          memset(out, 0xCD, n * 4 + 64);
          arms[a](k);
          for (size_t i = 0; i < n; ++i) {
            size_t expected_index = i;
            if (a == 3)
              expected_index = (i / kBlk) * kBlk + (i % 32) * 32 + (i % kBlk) / 32;
            const uint32_t want = a == 5 ? static_cast<uint32_t>(seq.mins[i / kBlk]) +
                                               static_cast<uint32_t>(i % kBlk)
                                         : static_cast<uint32_t>(values[expected_index]);
            if (static_cast<uint32_t>(out[i]) != want) {
              fprintf(stderr, "MISMATCH %s/%s arm=%s copy=%zu at %zu\n", ds.name, pt.name,
                      kArm[a], k, i);
              return 1;
            }
          }
          const auto* guard = reinterpret_cast<const uint8_t*>(out + n);
          for (size_t i = 0; i < 64; ++i)
            if (guard[i] != 0xCD) {
              fprintf(stderr, "output overrun %s/%s arm=%s\n", ds.name, pt.name, kArm[a]);
              return 1;
            }
        }
      }
      if (options.verify) continue;
      const size_t iters = std::max(visits, std::max<size_t>(3, options.bytes / (n * 4)));
      const auto stats = bench::Measure(arms, kArms, prepare, visits, iters, n * 4,
                                        options, rng, raw, ds.name, pt.name, kArm);
      double medians[kArms];
      double max_cv = 0;
      for (int a = 0; a < kArms; ++a) {
        medians[a] = stats[a].median;
        max_cv = std::max(max_cv, stats[a].cv);
      }
      noisy |= max_cv > 0.05;
      Row row;
      row.dataset = ds.name;
      row.point = pt.name;
      row.n = n;
      for (int a = 0; a < kArms; ++a) row.gibs[a] = medians[a];
      row.avg_w = avg_w;
      row.src_mib = src_mib;
      row.dst_mib = dst_mib;
      row.cr = static_cast<double>(n) * 4 / static_cast<double>(fb_len);
      rows.push_back(row);

      printf(
          "%-22s %-9s %5.1f %5.2f %8.2f %8.2f | %8.1f %8.1f %8.1f %8.1f %8.1f"
          " %8.1f | %7.2fx %7.2fx %7.2fx %7.2fx %7.0f%% %7.0f%%\n",
          ds.name, pt.name, avg_w, row.cr, src_mib, dst_mib, medians[0], medians[1],
          medians[2], medians[3], medians[4], medians[5], medians[3] / medians[0],
          medians[3] / medians[1], medians[3] / medians[2], medians[4] / medians[1],
          100 * medians[1] / medians[5], 100 * medians[2] / medians[5]);
      fflush(stdout);
      if (csv) {
        fprintf(csv,
                "%s,%s,%zu,%.2f,%.4f,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%"
                "d,%d,%u,%.9g,%.9g,%.9g,%.9g\n",
                ds.name, pt.name, n, avg_w, row.cr, src_mib, dst_mib, medians[0],
                medians[1], medians[2], medians[3], medians[4], medians[5], max_cv,
                medians[3] / medians[2],
                std::fabs(std::log(medians[3] / medians[2])) <= std::log(1.05) &&
                    max_cv <= 0.05,
                options.reps, options.seed,
                double(src_copies * seq.bytes.size()) / (1 << 20),
                double(src_copies * ib_len) / (1 << 20),
                double(src_copies * src_stride) / (1 << 20),
                double(dst_copies * out_stride) / (1 << 20));
        fflush(csv);
      }
    }
  }

  if (selected == 0) {
    fprintf(stderr, "dataset filter matched nothing\n");
    return 1;
  }
  if (options.verify) {
    printf("verified %zu datasets at all six points (no timings)\n", selected);
    bench::Close(csv);
    bench::Close(raw);
    return 0;
  }
  // --- geomeans per working set --------------------------------------------
  printf("%s\n", dashes.c_str());
  for (const Point& pt : kPoints) {
    double g[kArms] = {1, 1, 1, 1, 1, 1};
    int cnt = 0;
    for (const Row& r : rows) {
      if (strcmp(r.point, pt.name) != 0) continue;
      for (int a = 0; a < kArms; ++a) g[a] *= r.gibs[a];
      ++cnt;
    }
    if (cnt == 0) continue;
    double m[kArms];
    for (int a = 0; a < kArms; ++a) m[a] = std::pow(g[a], 1.0 / cnt);
    printf(
        "%-22s %-9s %5s %5s %8s %8s | %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f |"
        " %7.2fx %7.2fx %7.2fx %7.2fx %7.0f%% %7.0f%%   (n=%d)\n",
        "GEOMEAN", pt.name, "", "", "", "", m[0], m[1], m[2], m[3], m[4], m[5],
        m[3] / m[0], m[3] / m[1], m[3] / m[2], m[4] / m[1], 100 * m[1] / m[5],
        100 * m[2] / m[5], cnt);
  }

  // The validity check, stated as a pass/fail rather than left to the reader.
  printf("\nvalidity: fl_unpk/intlv must be 1.00x (same PackBlock/UnpackBlock)\n");
  bool validity_failed = noisy;
  if (noisy) printf("NOISY RUN: at least one arm has CV > 5%%\n");
  for (const Point& pt : kPoints) {
    double worst = 1.0;
    std::string worst_ds;
    int cnt = 0;
    double lg = 0;
    for (const Row& r : rows) {
      if (strcmp(r.point, pt.name) != 0) continue;
      const double q = r.gibs[3] / r.gibs[2];
      lg += std::log(q);
      ++cnt;
      if (std::fabs(std::log(q)) > std::fabs(std::log(worst))) {
        worst = q;
        worst_ds = r.dataset;
      }
    }
    if (cnt == 0) continue;
    const double gm = std::exp(lg / cnt);
    const bool pass = std::fabs(std::log(worst)) <= std::log(1.05);
    printf("  %-9s geomean %.3fx   worst %.3fx on %-20s  %s\n", pt.name, gm, worst,
           worst_ds.c_str(), pass ? "PASS" : "FAIL");
    if (!pass) validity_failed = true;
  }
  if (validity_failed) {
    printf(
        "\nVALIDITY FAILED: fl_unpk and intlv run the same kernel over the same\n"
        "byte count and must tie within 5%%. A point where they do not is\n"
        "inconclusive: investigate timing noise, code generation and placement.\n");
  }
  bench::Close(csv);
  bench::Close(raw);
  return validity_failed ? 2 : 0;
}
