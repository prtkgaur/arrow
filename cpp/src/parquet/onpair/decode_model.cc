// Fits the decode side of the selector's objective.
//
// The selector has to compare candidates at encode time, when it can afford to
// train nine dictionaries but not to time nine decodes on the whole column. So it
// needs a *predictor* built out of quantities training already produced. This
// driver dumps the grid the predictor is fitted against: every budget on every
// corpus, with the two candidate predictors (code count, dictionary footprint)
// beside the measured time.
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

}  // namespace

int main(int argc, char** argv) {
  std::string dir = bench::CorpusDir(argc, argv);
  std::printf("corpus,budget,width,tokens,dict_bytes,stride_bytes,codes,stored,raw,dec_sec\n");
  for (const auto& path : bench::CorpusFiles(dir)) {
    bench::Corpus c = bench::ReadCorpus(path);
    if (c.offsets.size() < 2) continue;
    double th = bench::ThresholdFor(c.name);
    for (uint8_t b = 8; b <= 16; ++b) {
      op::Config cfg;
      cfg.max_dict_bits = b;
      cfg.threshold_fraction = th;
      cfg.seed = 42;
      cfg.prune_absent_literals = true;
      op::Column col =
          op::Compress(c.bytes.data(), c.raw_bytes(), c.offsets.data(), c.n_rows(), cfg);
      size_t width = IndexBits(col.dict.num_tokens());
      std::vector<uint32_t> cw(col.codes.begin(), col.codes.end());
      std::vector<uint8_t> packed = op::PackValues(cw.data(), cw.size(), width);
      std::vector<uint8_t> out(op::DecodedLen(col) + c.raw_bytes() + op::kDecodePadding, 0);
      std::vector<double> secs;
      for (int it = 0; it < bench::kDecodeIters; ++it) {
        auto t0 = bench::Clock::now();
        size_t w =
            op::DecompressPacked(col.dict, packed.data(), col.codes.size(), width, out.data());
        secs.push_back(std::chrono::duration<double>(bench::Clock::now() - t0).count());
        asm volatile("" ::"r"(w) : "memory");
        if (w != c.raw_bytes() || std::memcmp(out.data(), c.bytes.data(), c.raw_bytes()) != 0) {
          std::fprintf(stderr, "ROUNDTRIP MISMATCH %s b%u\n", c.name.c_str(), b);
          return 1;
        }
      }
      // The decode-side view is a fixed 16-byte stride per token, so the working set
      // the kernel actually walks is 16 * tokens plus the parallel length array --
      // not the logical byte total, which is what the stored form charges.
      std::printf("%s,%u,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%.9f\n", c.name.c_str(), b, width,
                  col.dict.num_tokens(), col.dict.logical_bytes(), col.dict.num_tokens() * 17,
                  col.codes.size(), StoredBytes(col, c), c.raw_bytes(),
                  bench::Median(std::move(secs)));
      std::fflush(stdout);
    }
  }
  return 0;
}
