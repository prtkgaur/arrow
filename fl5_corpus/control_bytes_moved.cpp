// How many INPUT bytes does each arm actually read per value? If the sequential
// and interleaved payloads are not the same size, then at any point where the
// run is bandwidth-bound the arms' throughput ratio is the byte ratio and
// carries no information about the layout.
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include "arrow/util/fastlanes/interleaved_pfor.h"
#include "corpus_generators.h"

// Everything below is private to this file. One study per translation unit,
// so a study's kernels are compiled exactly as they were when it was a
// standalone binary, and two studies can hold the same name for different
// things.
namespace {

namespace fl = arrow::util::fastlanes;
using fl::InterleavedPforOrder;
static constexpr size_t kBlk = 1024;

static size_t SeqBytes(const std::vector<int32_t>& v) {
  size_t off = 0;
  for (size_t b = 0; b + kBlk <= v.size(); b += kBlk) {
    int32_t mn = v[b], mx = v[b];
    for (size_t t = 1; t < kBlk; ++t) { mn = std::min(mn, v[b+t]); mx = std::max(mx, v[b+t]); }
    uint32_t span = (uint32_t)mx - (uint32_t)mn, w = 0;
    while (w < 32 && (span >> w) != 0) ++w;
    off += w * kBlk / 8;
  }
  return off;
}

struct D { const char* name; std::vector<int32_t>(*g)(int64_t); };
#define D_(N) {#N, corpus::Gen##N}
#define DS_(N) {#N, corpus::delta_shapes::Gen##N<int32_t>}
static const D kD[] = {
  D_(ClientIP), D_(UrlRegionID), D_(CounterID), D_(EventDate), D_(EventTime),
  D_(GoodEvent), D_(HID), D_(HitColor), D_(IPNetworkID), D_(JavaEnable), D_(OS),
  D_(Resolution), D_(TrafficSourceID), D_(UserAgent),
  D_(TpcdsSoldDateSk), D_(TpcdsStoreSk), D_(TpcdsItemSk), D_(TpcdsQuantity),
  D_(TpcdsCustomerSk), D_(TpcdsExtSalesPrice), D_(TpcdsNetProfit), D_(TpcdsDYear),
  D_(TpchLQuantity), D_(TpchLExtendedPrice), D_(TpchLDiscount), D_(TpchLShipDate),
  D_(TaxiPickupUnixTime), D_(TaxiTripDistanceX100), D_(TaxiFareCents),
  D_(SortedUnixTime), D_(SortedKeyDups), D_(MonotoneRowId), D_(NearSortedUnixTime),
  DS_(TrendJitter), DS_(Sawtooth), {"MeasurementSeries", corpus::delta_shapes::GenMeasurement<int32_t>},
  DS_(SortedKeys), DS_(SensorDropouts), DS_(IdsWithGaps), DS_(RandomWalk),
  DS_(EventMillis), DS_(LowSentinel), DS_(Bimodal),
};
}  // namespace


int RunBytesMovedControl(int, char**) {
  const size_t n = 8192ull * 1024;
  printf("n=%zu (32 MiB of output)\n", n);
  printf("%-22s %12s %12s %8s\n", "dataset", "seq bytes", "intlv bytes", "intlv/seq");
  double lg = 0; int c = 0; double worst = 1;  const char* worst_n = "";
  std::vector<uint8_t> ib(fl::InterleavedPforMaxEncodedSize(n));
  for (const D& d : kD) {
    auto v = d.g((int64_t)n);
    size_t sb = SeqBytes(v);
    size_t il = fl::InterleavedPforEncode<InterleavedPforOrder::kFileOrder>(v.data(), n, ib.data());
    double r = (double)il / (double)sb;
    lg += std::log(r); ++c;
    if (r > worst) { worst = r; worst_n = d.name; }
    printf("%-22s %12zu %12zu %7.3fx%s\n", d.name, sb, il, r, r > 1.15 ? "  <<" : "");
  }
  printf("\ngeomean intlv/seq input bytes = %.3fx (n=%d), worst %.3fx on %s\n",
         std::exp(lg / c), c, worst, worst_n);
  return 0;
}
