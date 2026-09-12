// Separates two things that MiB/s of output conflates: how fast the kernel retires
// codes, and how many output bytes each code carries. A narrow rung can look slow
// per output byte purely because its tokens are shorter, or because the bit-unpack
// path for that width is worse. Only the second is a bug.
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "parquet/onpair/bench_common.h"
#include "parquet/onpair/onpair.h"

namespace op = parquet::onpair;
using bench::BitPackedBytes;
using bench::BitWidth;
using bench::IndexBits;

namespace {

size_t StoredBytes(const op::Column& col, const bench::Corpus& c) {
  size_t db = col.dict.logical_bytes();
  return db + BitPackedBytes(col.dict.offsets.size(), std::max<size_t>(1, BitWidth(db))) +
         BitPackedBytes(col.codes.size(), IndexBits(col.dict.num_tokens())) + c.len_array_bytes();
}

struct Row {
  size_t width = 0, tokens = 0, codes = 0, stored = 0;
  double mibs = 0, mcodes = 0;
};

Row Measure(const bench::Corpus& c, uint8_t bits, double th, bool prune) {
  op::Config cfg;
  cfg.max_dict_bits = bits;
  cfg.threshold_fraction = th;
  cfg.prune_absent_literals = prune;
  op::Column col = op::Compress(c.bytes.data(), c.raw_bytes(), c.offsets.data(), c.n_rows(), cfg);
  Row r;
  r.width = IndexBits(col.dict.num_tokens());
  r.tokens = col.dict.num_tokens();
  r.codes = col.codes.size();
  r.stored = StoredBytes(col, c);
  std::vector<uint32_t> cw(col.codes.begin(), col.codes.end());
  std::vector<uint8_t> packed = op::PackValues(cw.data(), cw.size(), r.width);
  std::vector<uint8_t> out(op::DecodedLen(col) + c.raw_bytes() + op::kDecodePadding, 0);
  std::vector<double> mib, mc;
  for (int it = 0; it < bench::kDecodeIters; ++it) {
    auto t0 = bench::Clock::now();
    size_t w = op::DecompressPacked(col.dict, packed.data(), r.codes, r.width, out.data());
    double dt = std::chrono::duration<double>(bench::Clock::now() - t0).count();
    asm volatile("" ::"r"(w) : "memory");
    mib.push_back(bench::Mib(c.raw_bytes()) / dt);
    mc.push_back(static_cast<double>(r.codes) / dt / 1e6);
  }
  r.mibs = bench::Median(std::move(mib));
  r.mcodes = bench::Median(std::move(mc));
  return r;
}

const char* kOnly[] = {"l_shipmode", "c_mktsegment", "p_type", "c_name", "sha256_hex", "uuid_v4"};
bool Wanted(const std::string& n) {
  for (const char* k : kOnly)
    if (n.find(k) != std::string::npos) return true;
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  std::string dir = bench::CorpusDir(argc, argv);
  std::printf("%-22s %-7s %6s %8s %11s %8s %9s %10s %8s\n", "corpus", "variant", "width", "tokens",
              "codes", "B/code", "ratio", "dec MiB/s", "Mcode/s");
  for (const auto& path : bench::CorpusFiles(dir)) {
    bench::Corpus c = bench::ReadCorpus(path);
    if (c.offsets.size() < 2 || !Wanted(c.name)) continue;
    double th = bench::ThresholdFor(c.name);
    double raw = static_cast<double>(c.raw_bytes());
    // Every rung, so the shape of the trade is visible rather than just its endpoints.
    for (uint8_t b = 8; b <= 16; ++b) {
      Row r = Measure(c, b, th, /*prune=*/true);
      char tag[16];
      std::snprintf(tag, sizeof(tag), "b%u", b);
      std::printf("%-22s %-7s %5zub %8zu %11zu %8.2f %8.3fx %10.0f %8.1f\n", c.name.c_str(), tag,
                  r.width, r.tokens, r.codes, raw / static_cast<double>(r.codes), raw / r.stored,
                  r.mibs, r.mcodes);
    }
    std::fflush(stdout);
  }
  return 0;
}
