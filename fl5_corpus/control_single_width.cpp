// Decomposes each delta scheme's bits/value into packed payload and side data,
// and prints the per-block bit widths, so a size difference between the layouts
// can be attributed to the width the residuals need rather than to any padding.
// All three schemes pack densely -- the continuous arm LSB-first across byte
// boundaries, both container arms through the same PackBlock<W>, which fills
// every bit of each lane's word stream and lets a value straddle words -- so
// payload bytes are exactly ceil(n * W / 8) with nothing wasted. This program
// exists to show that, not to assume it.
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "arrow/util/fastlanes/lane_delta.h"
#include "arrow/util/fastlanes/transposed_delta.h"
#include "corpus_generators.h"

// Everything below is private to this file. One study per translation unit,
// so a study's kernels are compiled exactly as they were when it was a
// standalone binary, and two studies can hold the same name for different
// things.
namespace {


namespace fl = arrow::util::fastlanes;
constexpr size_t kBlk = 1024;

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


static std::string Hist(const std::vector<uint8_t>& w) {
  std::map<int, int> h;
  for (uint8_t x : w) ++h[x];
  std::string s;
  char buf[32];
  for (auto& kv : h) {
    snprintf(buf, sizeof(buf), "%dx%d ", kv.second, kv.first);
    s += buf;
  }
  return s;
}

// The continuous arm's per-block width: bits(max signed delta - min signed delta).
static std::vector<uint8_t> SeqWidths(const std::vector<int32_t>& v) {
  std::vector<uint8_t> out(v.size() / kBlk);
  for (size_t b = 0; b < out.size(); ++b) {
    uint32_t mn = 0, mx = 0;
    for (size_t t = 0; t < kBlk; ++t) {
      const size_t i = b * kBlk + t;
      const uint32_t prev = i == 0 ? 0u : static_cast<uint32_t>(v[i - 1]);
      const uint32_t d = static_cast<uint32_t>(v[i]) - prev;
      if (t == 0) {
        mn = mx = d;
      } else {
        if (static_cast<int32_t>(d) < static_cast<int32_t>(mn)) mn = d;
        if (static_cast<int32_t>(d) > static_cast<int32_t>(mx)) mx = d;
      }
    }
    uint32_t span = mx - mn, w = 0;
    while (w < 32 && (span >> w) != 0) ++w;
    out[b] = static_cast<uint8_t>(w);
  }
  return out;
}

}  // namespace


int RunSingleWidthControl(int argc, char** argv) {
  const size_t n = argc > 1 ? strtoull(argv[1], nullptr, 10) : 1024u * 1024;
  const size_t nblocks = n / kBlk;
  printf("n=%zu (%zu blocks)\n\n", n, nblocks);

  for (size_t d = 0; d < kNumDatasets; ++d) {
    const Dataset& ds = kDatasets[d];
    std::vector<int32_t> v = ds.gen(n);

    std::vector<uint8_t> ld(fl::LaneDeltaMaxEncodedSize(n));
    std::vector<uint8_t> td(fl::TransposedMaxEncodedSize(n));
    const size_t ld_len =
        fl::LaneDeltaEncode<fl::LaneDeltaOrder::kInterleaved>(v.data(), n, ld.data());
    const size_t td_len = fl::TransposedDeltaEncode<fl::TransposedBaseCoding::kPacked>(
        v.data(), n, td.data());

    const std::vector<uint8_t> ws = SeqWidths(v);
    const uint8_t* lw = ld.data() + fl::kLanes * sizeof(uint32_t);
    std::vector<uint8_t> wl(lw, lw + nblocks);
    std::vector<uint8_t> wt(td.data(), td.data() + nblocks);

    auto sum = [](const std::vector<uint8_t>& w) {
      size_t s = 0;
      for (uint8_t x : w) s += x;
      return s;
    };
    // Payload = the packed residuals alone; side = everything else the scheme
    // stores (seeds, bases, per-block widths and minima).
    const double pay_s = static_cast<double>(sum(ws)) / nblocks;
    const double pay_l = static_cast<double>(sum(wl)) / nblocks;
    const double pay_t = static_cast<double>(sum(wt)) / nblocks;
    const double tot_l = static_cast<double>(ld_len) * 8 / n;
    const double tot_t = static_cast<double>(td_len) * 8 / n;

    printf("%-22s payload bits/value: seq %5.2f  int %5.2f  fl %5.2f   "
           "side: int %+.3f  fl %+.3f\n",
           ds.name, pay_s, pay_l, pay_t, tot_l - pay_l, tot_t - pay_t);
    printf("%-22s   widths  seq [%s]\n", "", Hist(ws).c_str());
    printf("%-22s           int [%s]\n", "", Hist(wl).c_str());
    printf("%-22s           fl  [%s]\n", "", Hist(wt).c_str());
  }
  return 0;
}
