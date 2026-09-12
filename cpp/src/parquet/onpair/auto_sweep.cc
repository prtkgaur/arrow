// Full-column dictionary-budget sweep for OnPair.
//
// Reproduces the OnPair16 and OnPair-auto rows of fsst_onpair_benchmark without
// linking FSST, zstd or lz4, so the width ladder can be re-run on its own. The
// size accounting is OnPairSize copied verbatim from that binary -- if the two
// ever disagree on a ratio, this file is the one that is wrong.
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
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

size_t OnPairSize(const op::Column& col, const bench::Corpus& c) {
  size_t db = col.dict.logical_bytes();
  return db + BitPackedBytes(col.dict.offsets.size(), std::max<size_t>(1, BitWidth(db))) +
         BitPackedBytes(col.codes.size(), IndexBits(col.dict.num_tokens())) + c.len_array_bytes();
}

// Round-trip through the packed code stream at the stored width. A width that
// stores fewer bytes but cannot reproduce the input is not a candidate.
void VerifyRoundTrip(const op::Column& col, const bench::Corpus& c, size_t bits) {
  std::vector<uint32_t> cw(col.codes.begin(), col.codes.end());
  std::vector<uint8_t> packed = op::PackValues(cw.data(), cw.size(), bits);
  std::vector<uint8_t> out(op::DecodedLen(col) + op::kDecodePadding, 0);
  size_t w = op::DecompressPacked(col.dict, packed.data(), col.codes.size(), bits, out.data());
  if (w != c.raw_bytes() || std::memcmp(out.data(), c.bytes.data(), c.raw_bytes()) != 0) {
    std::fprintf(stderr, "ROUNDTRIP MISMATCH %s at %zub (w=%zu raw=%zu)\n", c.name.c_str(), bits, w,
                 c.raw_bytes());
    std::abort();
  }
}

double DecodeMibs(const op::Column& col, const bench::Corpus& c, size_t bits) {
  std::vector<uint32_t> cw(col.codes.begin(), col.codes.end());
  std::vector<uint8_t> packed = op::PackValues(cw.data(), cw.size(), bits);
  size_t cap = op::DecodedLen(col) + op::kDecodePadding;
  std::vector<double> r;
  for (int it = 0; it < bench::kDecodeIters; ++it) {
    std::vector<uint8_t> out(cap, 0);
    auto t0 = bench::Clock::now();
    size_t w = op::DecompressPacked(col.dict, packed.data(), col.codes.size(), bits, out.data());
    double dt = std::chrono::duration<double>(bench::Clock::now() - t0).count();
    asm volatile("" ::"r"(w) : "memory");
    r.push_back(bench::Mib(c.raw_bytes()) / dt);
  }
  return bench::Median(std::move(r));
}

constexpr uint8_t kLoDefault = 9;

}  // namespace

int main(int argc, char** argv) {
  std::string dir = bench::CorpusDir(argc, argv);
  uint8_t lo = kLoDefault;
  if (const char* e = std::getenv("ONPAIR_MIN_BITS")) lo = static_cast<uint8_t>(std::atoi(e));
  const uint8_t hi = 16;

  std::printf("# budget ladder %u..%u\n", lo, hi);
  std::printf("%-30s %8s %9s %6s %6s %7s %9s %10s   %s\n", "corpus", "raw MiB", "op16 ratio",
              "budget", "width", "tokens", "auto ratio", "auto dec", "per-budget ratio");
  for (const auto& path : bench::CorpusFiles(dir)) {
    bench::Corpus c = bench::ReadCorpus(path);
    if (c.offsets.size() < 2) continue;
    double threshold = bench::ThresholdFor(c.name);

    uint8_t best_b = lo;
    size_t best_sz = SIZE_MAX, best_tok = 0;
    std::vector<double> ratios;
    for (uint8_t b = lo; b <= hi; ++b) {
      op::Config cfg{b, threshold, 42};
      op::Column col =
          op::Compress(c.bytes.data(), c.raw_bytes(), c.offsets.data(), c.n_rows(), cfg);
      size_t sz = OnPairSize(col, c);
      VerifyRoundTrip(col, c, IndexBits(col.dict.num_tokens()));
      ratios.push_back(static_cast<double>(c.raw_bytes()) / static_cast<double>(sz));
      if (sz < best_sz) {
        best_sz = sz;
        best_b = b;
        best_tok = col.dict.num_tokens();
      }
    }

    op::Config cfg{best_b, threshold, 42};
    op::Column col = op::Compress(c.bytes.data(), c.raw_bytes(), c.offsets.data(), c.n_rows(), cfg);
    size_t stored_bits = IndexBits(col.dict.num_tokens());
    double auto_ratio = static_cast<double>(c.raw_bytes()) / static_cast<double>(best_sz);
    double op16 = ratios[hi - lo];

    std::printf("%-30s %8.2f %9.3fx %5ub %5zub %7zu %9.3fx %10.0f  ", c.name.c_str(),
                bench::Mib(c.raw_bytes()), op16, best_b, stored_bits, best_tok, auto_ratio,
                DecodeMibs(col, c, stored_bits));
    for (double r : ratios) std::printf(" %.3f", r);
    std::printf("\n");
    std::fflush(stdout);
  }
  return 0;
}
