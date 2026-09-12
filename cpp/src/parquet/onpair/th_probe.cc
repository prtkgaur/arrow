// Why the narrowest rung decodes slower than the rung above it.
//
// At budget 8 the trainer stops well short of its capacity on low-cardinality
// columns -- tpch_l_shipmode settles on 36 tokens with ~249 pair slots free. Short
// tokens mean more codes per output byte, which is a decode cost, so the ratio win
// at that rung may be paid for by leaving the dictionary half empty rather than by
// anything intrinsic to an 8-bit code. This sweeps the training threshold at budget
// 8 to see whether filling the dictionary recovers the decode.
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

struct Trial {
  op::Column col;
  size_t stored = 0, width = 0;
};

Trial RunOne(const bench::Corpus& c, uint8_t bits, double th, bool prune) {
  op::Config cfg;
  cfg.max_dict_bits = bits;
  cfg.threshold_fraction = th;
  cfg.prune_absent_literals = prune;
  Trial t;
  t.col = op::Compress(c.bytes.data(), c.raw_bytes(), c.offsets.data(), c.n_rows(), cfg);
  t.stored = StoredBytes(t.col, c);
  t.width = IndexBits(t.col.dict.num_tokens());
  return t;
}

double DecodeMibs(const op::Column& col, const bench::Corpus& c, size_t width) {
  std::vector<uint32_t> cw(col.codes.begin(), col.codes.end());
  std::vector<uint8_t> packed = op::PackValues(cw.data(), cw.size(), width);
  size_t cap = op::DecodedLen(col) + c.raw_bytes() + op::kDecodePadding;
  std::vector<uint8_t> out(cap, 0);
  size_t w = op::DecompressPacked(col.dict, packed.data(), col.codes.size(), width, out.data());
  if (w != c.raw_bytes() || std::memcmp(out.data(), c.bytes.data(), c.raw_bytes()) != 0) {
    std::fprintf(stderr, "ROUNDTRIP MISMATCH %s\n", c.name.c_str());
    std::abort();
  }
  std::vector<double> r;
  for (int it = 0; it < bench::kDecodeIters; ++it) {
    auto t0 = bench::Clock::now();
    size_t n = op::DecompressPacked(col.dict, packed.data(), col.codes.size(), width, out.data());
    double dt = std::chrono::duration<double>(bench::Clock::now() - t0).count();
    asm volatile("" ::"r"(n) : "memory");
    r.push_back(bench::Mib(c.raw_bytes()) / dt);
  }
  return bench::Median(std::move(r));
}

const char* kOnly[] = {"c_mktsegment", "l_shipmode", "o_orderpriority", "p_brand",
                       "p_container",  "p_type",     "s_name",          "c_name",
                       "sha256_hex",   "uuid_v4"};

bool Wanted(const std::string& n) {
  for (const char* k : kOnly)
    if (n.find(k) != std::string::npos) return true;
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  std::string dir = bench::CorpusDir(argc, argv);
  const double kThresholds[] = {-1.0, 0.15, 0.05, 0.02, 0.01, 0.004, 0.001};
  for (const auto& path : bench::CorpusFiles(dir)) {
    bench::Corpus c = bench::ReadCorpus(path);
    if (c.offsets.size() < 2 || !Wanted(c.name)) continue;
    double dflt = bench::ThresholdFor(c.name);
    double raw = static_cast<double>(c.raw_bytes());

    // Baseline: the published ladder, 9..16 at the default threshold.
    Trial best;
    best.stored = SIZE_MAX;
    uint8_t bb = 9;
    for (uint8_t b = 9; b <= 16; ++b) {
      Trial t = RunOne(c, b, dflt, false);
      if (t.stored < best.stored) { best = std::move(t); bb = b; }
    }
    std::printf("%s  (default threshold %.3f)\n", c.name.c_str(), dflt);
    std::printf("  %-14s %5s %6s %8s %9s %9s\n", "variant", "bud", "width", "tokens", "ratio",
                "dec MiB/s");
    std::printf("  %-14s %4ub %5zub %8zu %8.3fx %9.0f\n", "autoA", bb, best.width,
                best.col.dict.num_tokens(), raw / best.stored, DecodeMibs(best.col, c, best.width));
    for (double th : kThresholds) {
      double use = th < 0 ? dflt : th;
      Trial t = RunOne(c, 8, use, true);
      char tag[32];
      std::snprintf(tag, sizeof(tag), "b8 th=%.3f", use);
      std::printf("  %-14s %4ub %5zub %8zu %8.3fx %9.0f\n", tag, 8, t.width,
                  t.col.dict.num_tokens(), raw / t.stored, DecodeMibs(t.col, c, t.width));
    }
    std::fflush(stdout);
  }
  return 0;
}
