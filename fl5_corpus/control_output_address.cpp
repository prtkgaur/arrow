// Is the corpus harness's DRAM-point validity failure an input-alignment
// artifact? intlv and fl_unpk call the same PackBlock/UnpackBlock and must tie.
// The harness shares one aligned OUTPUT buffer but gives each arm its own
// std::vector INPUT, so input-address-mod-4096 is a free variable. Run the same
// pair twice: once with independent vectors, once from one aligned arena at
// identical offsets.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>
#include "arrow/util/fastlanes/interleaved_pfor.h"
#include "corpus_generators.h"

// Everything below is private to this file. One study per translation unit,
// so a study's kernels are compiled exactly as they were when it was a
// standalone binary, and two studies can hold the same name for different
// things.
namespace {

namespace fl = arrow::util::fastlanes;
using fl::InterleavedPforOrder;
using Clock = std::chrono::steady_clock;

static uint8_t* Arena(size_t n) {
  void* p = nullptr;
  if (posix_memalign(&p, 4096, n + 4096) != 0) abort();
  memset(p, 0, n + 4096);
  return static_cast<uint8_t*>(p);
}

}  // namespace


int RunOutputAddressControl(int argc, char** argv) {
  const size_t n = (argc > 1) ? strtoull(argv[1], nullptr, 10) : 8192ull * 1024;
  auto values = corpus::delta_shapes::GenEventMillis<int32_t>((int64_t)n);
  const size_t cap = fl::InterleavedPforMaxEncodedSize(n);

  int32_t* out = reinterpret_cast<int32_t*>(Arena(n * 4));

  // (A) exactly what the harness does: two independent vectors.
  std::vector<uint8_t> vi(cap), vf(cap);
  size_t li = fl::InterleavedPforEncode<InterleavedPforOrder::kFileOrder>(values.data(), n, vi.data());
  size_t lf = fl::InterleavedPforEncode<InterleavedPforOrder::kFlOrder>(values.data(), n, vf.data());

  // (B) one aligned arena, both payloads at the same offset mod 4096.
  const size_t stride = (cap + 4095) / 4096 * 4096;
  uint8_t* arena = Arena(2 * stride);
  memcpy(arena, vi.data(), li);
  memcpy(arena + stride, vf.data(), lf);

  printf("n=%zu  wire: intlv=%zu fl=%zu %s\n", n, li, lf, li == lf ? "(equal)" : "(DIFFER)");
  printf("vector addrs: intlv=%p (mod4096=%4zu)  fl=%p (mod4096=%4zu)\n",
         (void*)vi.data(), (uintptr_t)vi.data() % 4096,
         (void*)vf.data(), (uintptr_t)vf.data() % 4096);

  const size_t iters = std::max<size_t>(8, (1ull << 30) / (n * 4));
  auto run = [&](const uint8_t* src, bool fl_order) {
    double best = 0;
    for (int rep = 0; rep < 7; ++rep) {
      if (fl_order) fl::InterleavedPforDecode<InterleavedPforOrder::kFlOrderRaw>(src, n, out);
      else fl::InterleavedPforDecode<InterleavedPforOrder::kFileOrder>(src, n, out);
      auto t0 = Clock::now();
      for (size_t it = 0; it < iters; ++it) {
        if (fl_order) fl::InterleavedPforDecode<InterleavedPforOrder::kFlOrderRaw>(src, n, out);
        else fl::InterleavedPforDecode<InterleavedPforOrder::kFileOrder>(src, n, out);
      }
      double s = std::chrono::duration<double>(Clock::now() - t0).count();
      best = std::max(best, (double)iters * n * 4 / s / (1024.0*1024*1024));
    }
    return best;
  };
  double a_i = run(vi.data(), false), a_f = run(vf.data(), true);
  double b_i = run(arena, false),     b_f = run(arena + stride, true);
  printf("\n%-28s %8s %8s %10s\n", "input placement", "intlv", "fl_unpk", "fl/intlv");
  printf("%-28s %8.1f %8.1f %9.3fx   <- must be 1.000x\n", "independent vectors (harness)", a_i, a_f, a_f/a_i);
  printf("%-28s %8.1f %8.1f %9.3fx\n", "one arena, matched offsets", b_i, b_f, b_f/b_i);
  return 0;
}
