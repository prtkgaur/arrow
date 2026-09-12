// Both dictionary-budget ladders in one process.
//
//   op16    budget 16, full byte residency          -- cross-check vs RESULTS_FSST16.txt
//   auto-A  best of 9..16, full byte residency      -- the published OnPair-auto
//   auto-B  best of 8..16, occurring bytes only     -- the extended ladder
//
// A and B share this binary so their decode figures are comparable; ratios from a
// separate run are comparable across binaries but decode numbers are not.
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

// Copied verbatim from fsst_onpair_benchmark's OnPairSize, so the reproduced ratios
// are comparable to the published ones rather than merely similar.
size_t StoredBytes(const op::Column& col, const bench::Corpus& c) {
  size_t db = col.dict.logical_bytes();
  return db + BitPackedBytes(col.dict.offsets.size(), std::max<size_t>(1, BitWidth(db))) +
         BitPackedBytes(col.codes.size(), IndexBits(col.dict.num_tokens())) + c.len_array_bytes();
}

struct Trial {
  op::Column col;
  size_t stored = 0;
  size_t width = 0;
};

Trial RunOne(const bench::Corpus& c, uint8_t bits, double threshold, bool prune) {
  op::Config cfg;
  cfg.max_dict_bits = bits;
  cfg.threshold_fraction = threshold;
  cfg.seed = 42;
  cfg.prune_absent_literals = prune;
  Trial t;
  t.col = op::Compress(c.bytes.data(), c.raw_bytes(), c.offsets.data(), c.n_rows(), cfg);
  t.stored = StoredBytes(t.col, c);
  t.width = IndexBits(t.col.dict.num_tokens());
  return t;
}

std::vector<uint8_t> PackCodes(const op::Column& col, size_t width) {
  std::vector<uint32_t> cw(col.codes.begin(), col.codes.end());
  return op::PackValues(cw.data(), cw.size(), width);
}

void Verify(const op::Column& col, const bench::Corpus& c, size_t width, const char* tag) {
  std::vector<uint8_t> packed = PackCodes(col, width);
  std::vector<uint8_t> out(op::DecodedLen(col) + c.raw_bytes() + op::kDecodePadding, 0);
  size_t w = op::DecompressPacked(col.dict, packed.data(), col.codes.size(), width, out.data());
  if (w != c.raw_bytes() || std::memcmp(out.data(), c.bytes.data(), c.raw_bytes()) != 0) {
    std::fprintf(stderr, "ROUNDTRIP MISMATCH %s/%s at %zub (w=%zu raw=%zu)\n", c.name.c_str(), tag,
                 width, w, c.raw_bytes());
    std::abort();
  }
}

double DecodeMibs(const op::Column& col, const bench::Corpus& c, size_t width) {
  std::vector<uint8_t> packed = PackCodes(col, width);
  size_t cap = op::DecodedLen(col) + c.raw_bytes() + op::kDecodePadding;
  std::vector<double> r;
  for (int it = 0; it < bench::kDecodeIters; ++it) {
    std::vector<uint8_t> out(cap, 0);
    auto t0 = bench::Clock::now();
    size_t w = op::DecompressPacked(col.dict, packed.data(), col.codes.size(), width, out.data());
    double dt = std::chrono::duration<double>(bench::Clock::now() - t0).count();
    asm volatile("" ::"r"(w) : "memory");
    r.push_back(bench::Mib(c.raw_bytes()) / dt);
  }
  return bench::Median(std::move(r));
}

// Best rung of [lo, 16] by stored bytes, verifying every candidate round-trips.
Trial BestRung(const bench::Corpus& c, uint8_t lo, double threshold, bool prune, uint8_t* out_budget,
               std::vector<double>* out_ratios) {
  Trial best;
  best.stored = SIZE_MAX;
  for (uint8_t b = lo; b <= 16; ++b) {
    Trial t = RunOne(c, b, threshold, prune);
    Verify(t.col, c, t.width, prune ? "B" : "A");
    if (out_ratios != nullptr) {
      out_ratios->push_back(static_cast<double>(c.raw_bytes()) / static_cast<double>(t.stored));
    }
    if (t.stored < best.stored) {
      best = std::move(t);
      *out_budget = b;
    }
  }
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  std::string dir = bench::CorpusDir(argc, argv);
  std::printf("%-30s %8s %8s %8s %6s %6s %8s %6s %6s %7s %8s %8s\n", "corpus", "raw MiB", "op16",
              "autoA", "Abud", "Awid", "autoB", "Bbud", "Bwid", "Btokens", "Adec", "Bdec");
  for (const auto& path : bench::CorpusFiles(dir)) {
    bench::Corpus c = bench::ReadCorpus(path);
    if (c.offsets.size() < 2) continue;
    double th = bench::ThresholdFor(c.name);

    uint8_t abud = 9, bbud = 8;
    Trial a = BestRung(c, 9, th, false, &abud, nullptr);
    Trial b = BestRung(c, 8, th, true, &bbud, nullptr);

    Trial op16 = RunOne(c, 16, th, false);
    double raw = static_cast<double>(c.raw_bytes());
    std::printf("%-30s %8.2f %7.3fx %7.3fx %5ub %5zub %7.3fx %5ub %5zub %7zu %8.0f %8.0f\n",
                c.name.c_str(), bench::Mib(c.raw_bytes()), raw / op16.stored, raw / a.stored, abud,
                a.width, raw / b.stored, bbud, b.width, b.col.dict.num_tokens(),
                DecodeMibs(a.col, c, a.width), DecodeMibs(b.col, c, b.width));
    std::fflush(stdout);
  }
  return 0;
}
