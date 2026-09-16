// Dispatch/granularity probe at fixed width; report directly, never use as a universal
// correction factor.
#include <arrow/util/bpacking_internal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>
#include <xsimd/xsimd.hpp>
#include "arrow/util/bpacking_dispatch_internal.h"
#include "arrow/util/bpacking_simd_internal.h"
#include "arrow/util/bpacking_simd_kernel_internal.h"
#include "timing.h"

namespace bp = arrow::internal::bpacking;
using Clock = std::chrono::steady_clock;
template <typename U, int W>
using KernelDefault = bp::Kernel<U, W, xsimd::default_arch>;

template <uint32_t W>
static void Run(size_t n, const char* point, const bench::Options& options,
                std::mt19937& order_rng, FILE* raw, double* a, double* b, double* c) {
  std::mt19937 rng(7 + W);
  const uint32_t span = (W == 32) ? 0xFFFFFFFFu : ((1u << W) - 1);
  std::vector<uint32_t> vals(n);
  for (auto& v : vals) v = rng() & span;
  std::vector<uint8_t> packed(n * W / 8 + 64, 0);
  {
    uint64_t acc = 0;
    int bits = 0;
    uint8_t* p = packed.data();
    for (size_t i = 0; i < n; ++i) {
      acc |= static_cast<uint64_t>(vals[i]) << bits;
      bits += W;
      while (bits >= 8) {
        *p++ = static_cast<uint8_t>(acc);
        acc >>= 8;
        bits -= 8;
      }
    }
    if (bits) *p = static_cast<uint8_t>(acc);
  }
  void* allocation;
  if (posix_memalign(&allocation, 4096, n * 4 + 4096) != 0) abort();
  uint32_t* out = static_cast<uint32_t*>(allocation);

  arrow::internal::UnpackOptions blk, all;
  blk.batch_size = 1024;
  blk.bit_width = W;
  all.batch_size = static_cast<int>(n);
  all.bit_width = W;
  all.max_read_bytes = static_cast<int>(packed.size());

  auto per_block = [&](size_t) {
    for (size_t i = 0; i < n; i += 1024) {
      blk.max_read_bytes = static_cast<int>(packed.size() - i * W / 8);
      arrow::internal::unpack_bias<uint32_t>(packed.data() + i * W / 8, out + i, blk, 5u);
    }
  };
  auto whole_buf = [&](size_t) {
    arrow::internal::unpack_bias<uint32_t>(packed.data(), out, all, 5u);
  };
  auto per_blk_ct = [&](size_t) {
    for (size_t i = 0; i < n; i += 1024) {
      blk.max_read_bytes = static_cast<int>(packed.size() - i * W / 8);
      bp::unpack_jump<KernelDefault, true>(packed.data() + i * W / 8, out + i, blk, 5u);
    }
  };

  std::function<void(size_t)> arms[] = {per_block, whole_buf, per_blk_ct};
  const char* names[] = {"per_block", "whole_buf", "per_blk_ct"};
  for (const auto& fn : arms) {
    std::fill(out, out + n, 0xCDCDCDCDu);
    fn(0);
    for (size_t i = 0; i < n; ++i)
      if (out[i] != vals[i] + 5u) {
        fprintf(stderr, "calibration mismatch at %zu width %u\n", i, W);
        std::exit(1);
      }
  }
  if (options.verify) {
    free(allocation);
    return;
  }
  const size_t iters = std::max<size_t>(3, options.bytes / (n * 4));
  const std::string width = std::to_string(W);
  auto stats = bench::Measure(
      arms, 3, [](int) {}, 1, iters, n * 4, options, order_rng, raw, width.c_str(), point,
      names);
  free(allocation);
  *a = stats[0].median;
  *b = stats[1].median;
  *c = stats[2].median;
}

int main(int argc, char** argv) {
  auto options = bench::Parse(argc, argv);
  if (options.filter != "all") {
    fprintf(stderr, "probe filter must be all\n");
    return 1;
  }
  std::mt19937 order_rng(options.seed);
  FILE* raw = bench::Open(options.csv);
  if (raw) fprintf(raw, "dataset,point,arm,repetition,order,iterations,seconds,gibs\n");
  printf("# median-of-%d; seed=%u; bounds extend through the padded payload\n",
         options.reps, options.seed);

  struct P {
    const char* nm;
    size_t n;
  };
  const P pts[] = {{"16KiB", 4096}, {"256KiB", 65536}, {"1MiB", 262144}};
  printf("%-5s %2s | %9s %9s %10s | %8s %8s\n", "point", "W", "per_block", "whole_buf",
         "per_blk_ct", "whole/blk", "ct/blk");
  for (const P& p : pts) {
    double g[3] = {1, 1, 1};
    int n = 0;
    double a = 1, b = 1, c = 1;
#define R(W)                                                                           \
  Run<W>(p.n, p.nm, options, order_rng, raw, &a, &b, &c);                              \
  if (!options.verify)                                                                 \
    printf("%-5s %2d | %9.1f %9.1f %10.1f | %7.2fx %7.2fx\n", p.nm, W, a, b, c, b / a, \
           c / a);                                                                     \
  g[0] *= a;                                                                           \
  g[1] *= b;                                                                           \
  g[2] *= c;                                                                           \
  ++n;
    R(0)
    R(1) R(4) R(7) R(11) R(12) R(18) R(20) R(24) R(31) R(32)
#undef R
        double m0 = std::pow(g[0], 1.0 / n),
               m1 = std::pow(g[1], 1.0 / n), m2 = std::pow(g[2], 1.0 / n);
    if (!options.verify)
      printf("%-5s gm | %9.1f %9.1f %10.1f | %7.2fx %7.2fx\n\n", p.nm, m0, m1, m2,
             m1 / m0, m2 / m0);
  }
  if (options.verify)
    printf("verified 11 widths at all three decoded sizes (no timings)\n");
  bench::Close(raw);
  return 0;
}
