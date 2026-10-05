// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "benchmark/benchmark.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <utility>

#include <xsimd/xsimd.hpp>

#include "arrow/util/bpacking_dispatch_internal.h"
#include "arrow/util/bpacking_internal.h"
#include "arrow/util/bpacking_scalar_internal.h"
#include "arrow/util/bpacking_simd_kernel_internal.h"
#include "arrow/util/fastlanes/fastlanes_kernels_internal.h"
#include "arrow/util/fastlanes/interleaved_pfor.h"
#include "arrow/util/fastlanes/transposed_delta.h"
#include "arrow/util/macros.h"

// Times the interleaved block decoder against the three sequential bit-unpack
// paths Arrow has, with the decoded footprint held fixed, so a ratio between
// two of them reflects the layout and not a difference in how much data each
// one moved.
//
// Both decoders read w bits per value and write whole elements, so at a given
// width and footprint their source and destination sizes agree exactly: a block
// of 1024 values occupies 128 * w packed bytes either way. The element type is
// the narrowest whole byte that holds w bits, which is what a reader decoding
// into a fixed-width column would choose, and which keeps the destination
// footprint from growing fourfold for a width that needs a single byte.

namespace arrow::util::fastlanes {

namespace {

// Arrow's sequential decoders may read a whole word past the last value they
// need, so every packed buffer carries this much slack past its last byte.
constexpr size_t kPackedTailSlack = 64;

// The narrowest whole-byte element that holds a w-bit value.
template <uint32_t w>
using NextWholeByte =
    std::conditional_t<(w <= 8), uint8_t,
                       std::conditional_t<(w <= 16), uint16_t, uint32_t>>;

// Every buffer is aligned the way Arrow aligns a column's buffers rather than
// the way new and vector do, which is to the element. The interleaved decoder
// stores a whole register at a time, and on a destination that is not
// register-aligned each of those stores straddles two cache lines and the row
// reads half speed. Leaving the alignment to the allocator makes a row's
// throughput depend on what the process allocated before it, a factor of two
// at one width on this ladder.
constexpr size_t kBufferAlignment = 64;

template <typename T>
struct AlignedFree {
  void operator()(T* p) const { std::free(p); }
};

template <typename T>
using AlignedArray = std::unique_ptr<T[], AlignedFree<T>>;

template <typename T>
AlignedArray<T> AllocateAligned(size_t num_elements) {
  const size_t bytes = (num_elements * sizeof(T) + kBufferAlignment - 1) /
                       kBufferAlignment * kBufferAlignment;
  auto* p = static_cast<T*>(std::aligned_alloc(kBufferAlignment, bytes));
  if (p == nullptr) std::abort();
  return AlignedArray<T>(p);
}

// Packed bytes are random rather than packed from known values: both decoders
// are data-independent at a fixed width, and a round trip is the unit test's
// job rather than the benchmark's.
// Filled eight bytes at a time: the largest footprints on the ladder are a
// gigabyte, and one draw per element made setup cost more than the measurement.
template <typename T>
void RandomFill(T* data, size_t n) {
  std::mt19937_64 rng(42);
  auto* bytes = reinterpret_cast<uint8_t*>(data);
  const size_t total = n * sizeof(T);
  size_t i = 0;
  for (; i + sizeof(uint64_t) <= total; i += sizeof(uint64_t)) {
    const uint64_t v = rng();
    std::memcpy(bytes + i, &v, sizeof(v));
  }
  for (; i < total; ++i) bytes[i] = static_cast<uint8_t>(rng());
}

// Both sequential decoders are driven a page at a time rather than once over
// the whole footprint, because one call over the whole footprint overflows
// them: batch_size is an int and the unpacker multiplies it by the bit width,
// so a 1 GiB footprint of bytes at three bits wraps past INT_MAX and the call
// returns having decoded nothing.
//
// The page is 16384 values, or the whole footprint when that is smaller. Call
// granularity is not free for this decoder and the size was measured rather
// than assumed: against a single call it reads 0.58x at 1024 values and 0.86x
// at 4096 for three bits into bytes, and 0.95x to 1.00x at 16384 across every
// width and footprint tried. A narrower page would slow the sequential
// decoders for a reason unrelated to the layout.
constexpr size_t kSequentialPageValues = 16384;

// Named for the same reason DecodePage is: a loop left in the registered
// lambda shares one optimization budget with every other width in this file.
template <typename T, uint32_t w, typename Fn>
ARROW_NOINLINE void DecodePagesSequential(const uint8_t* packed, T* out,
                                          size_t num_values, size_t page_values,
                                          Fn&& decode) {
  // The last page is short whenever the footprint is not a whole number of
  // pages, and it has to be decoded too: dropping it while still reporting the
  // whole footprint as processed inflates the row by the fraction left out, a
  // third at the 1.5-page footprint on this ladder.
  ::arrow::internal::UnpackOptions opts{.batch_size = static_cast<int>(page_values),
                               .bit_width = static_cast<int>(w)};
  for (size_t v = 0; v < num_values; v += page_values) {
    const size_t n = std::min(page_values, num_values - v);
    opts.batch_size = static_cast<int>(n);
    decode(packed + v * w / 8, out + v, opts);
  }
}

// The dispatched entry point Arrow ships. On this configuration it is built as
// AVX-512; BM_SequentialSimd below drives the same template with the
// instruction set pinned to AVX2, and the comment there gives the mechanism.
template <typename T, uint32_t w>
void BM_Sequential(benchmark::State& state, size_t out_bytes) {
  const size_t num_values = out_bytes / sizeof(T) / kBlockSize * kBlockSize;
  const size_t page_values = std::min(kSequentialPageValues, num_values);
  const size_t packed_bytes = num_values * w / 8;
  auto packed = AllocateAligned<uint8_t>(packed_bytes + kPackedTailSlack);
  RandomFill(packed.get(), packed_bytes + kPackedTailSlack);
  auto out = AllocateAligned<T>(num_values);
  const auto decode = [](const uint8_t* src, T* dst,
                         const ::arrow::internal::UnpackOptions& o) {
    ::arrow::internal::unpack<T>(src, dst, o);
  };

  for (auto _ : state) {
    DecodePagesSequential<T, w>(packed.get(), out.get(), num_values, page_values,
                                decode);
    benchmark::DoNotOptimize(out[0]);
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(static_cast<int64_t>(num_values * sizeof(T)) *
                          state.iterations());
  state.SetItemsProcessed(static_cast<int64_t>(num_values) * state.iterations());
}

// Arrow's vector kernel compiled in this translation unit, pinned to the AVX2
// instruction set rather than taken from xsimd's default.
//
// The dispatched entry point cannot stand in for it. libarrow is configured
// with ARROW_SIMD_LEVEL=AVX512, which appends the AVX-512 flag to the flags
// every file is compiled with, so inside bpacking_simd_256.cc -- the file whose
// kernels are named avx2 -- xsimd's default instruction set is AVX-512 and the
// kernels built there are the AVX-512 ones. Disassembling the shipped
// unpack_avx2<uint32_t> shows 113 zmm operands against 32 ymm, and 584
// vpextrb/vpinsrb instructions that move one byte at a time, which is the cost
// bpacking.cc cites when it declines to dispatch to AVX-512 at all. At three
// bits into bytes and at 17 to 24 bits into words that code reads around a
// twentieth of what the same template does when the kernel is AVX2, which is
// slower than scalar and would otherwise be read as the layout winning.
//
// The pin fixes which intrinsics the kernel uses, but it does not fix the code
// the compiler vectorizes around them, and this file still compiles under its
// own flags. -mprefer-vector-width reaches that code at eight widths: three,
// five, six and seven bits into bytes, and eleven, thirteen, fourteen and
// fifteen into words, compile with 128-bit operands where the adjacent widths
// carry none, and run three to eight times slower than those adjacent widths.
// A reading from this column therefore belongs to the flags this file was
// built with, and comparing one build against another needs a width that
// carries no 128-bit operands in either.
template <typename T, int w>
using Avx2Kernel = ::arrow::internal::bpacking::Kernel<T, w, xsimd::avx2>;

template <typename T, uint32_t w>
void BM_SequentialSimd(benchmark::State& state, size_t out_bytes) {
  const size_t num_values = out_bytes / sizeof(T) / kBlockSize * kBlockSize;
  const size_t page_values = std::min(kSequentialPageValues, num_values);
  const size_t packed_bytes = num_values * w / 8;
  auto packed = AllocateAligned<uint8_t>(packed_bytes + kPackedTailSlack);
  RandomFill(packed.get(), packed_bytes + kPackedTailSlack);
  auto out = AllocateAligned<T>(num_values);
  const auto decode = [](const uint8_t* src, T* dst,
                         const ::arrow::internal::UnpackOptions& o) {
    ::arrow::internal::bpacking::unpack_width<w, Avx2Kernel, false>(
        src, dst, o.batch_size, o.bit_offset, o.max_read_bytes, T{});
  };

  for (auto _ : state) {
    DecodePagesSequential<T, w>(packed.get(), out.get(), num_values, page_values,
                                decode);
    benchmark::DoNotOptimize(out[0]);
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(static_cast<int64_t>(num_values * sizeof(T)) *
                          state.iterations());
  state.SetItemsProcessed(static_cast<int64_t>(num_values) * state.iterations());
}

// The scalar kernel, called directly rather than through the dispatcher. At
// some bit widths the dispatched kernel is slower than this one; reporting
// both keeps a ratio against the vector kernel from being read as a property
// of the layout.
template <typename T, uint32_t w>
void BM_SequentialScalar(benchmark::State& state, size_t out_bytes) {
  const size_t num_values = out_bytes / sizeof(T) / kBlockSize * kBlockSize;
  const size_t page_values = std::min(kSequentialPageValues, num_values);
  const size_t packed_bytes = num_values * w / 8;
  auto packed = AllocateAligned<uint8_t>(packed_bytes + kPackedTailSlack);
  RandomFill(packed.get(), packed_bytes + kPackedTailSlack);
  auto out = AllocateAligned<T>(num_values);
  const auto decode = [](const uint8_t* src, T* dst,
                         const ::arrow::internal::UnpackOptions& o) {
    ::arrow::internal::bpacking::unpack_scalar<T>(src, dst, o);
  };

  for (auto _ : state) {
    DecodePagesSequential<T, w>(packed.get(), out.get(), num_values, page_values,
                                decode);
    benchmark::DoNotOptimize(out[0]);
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(static_cast<int64_t>(num_values * sizeof(T)) *
                          state.iterations());
  state.SetItemsProcessed(static_cast<int64_t>(num_values) * state.iterations());
}

// The decode loop lives in its own function per element width and bit width. A
// lambda body registered with the benchmark shares one optimization budget with
// every other width registered in this file, and the compiler exhausts it: the
// widths registered last then compile to a partly scalar loop, several times
// slower than the same kernel in its own function. Naming the loop
// gives each instantiation its own budget. It costs one call per page, which
// the sequential decoder pays as well.
template <typename T, uint32_t w>
ARROW_NOINLINE void DecodePage(const T* packed, T* out, size_t num_blocks) {
  using G = BlockGeometry<T>;
  for (size_t b = 0; b < num_blocks; ++b) {
    UnpackBlock<T, w, false>(packed + b * w * G::kLanes, out + b * kBlockSize);
  }
}

// The same packed block delivered in file order instead. UnpackBlock writes the
// values where the container holds them, which is the order InterleavedPforOrder
// calls kFlOrderRaw: a reader that wants a column in file order still owes the
// 32x32 permutation, and this is the decoder that pays it. Where the fused
// kernel exists the permutation happens in registers; otherwise the grid is
// materialized and read back, which is the fallback InterleavedPforDecode takes.
// Both are 32-bit only, so there is no file-order row at u8 or u16.
template <uint32_t w>
ARROW_NOINLINE void DecodePageFileOrder(const uint32_t* packed, int32_t* out,
                                        size_t num_blocks) {
  using G = BlockGeometry<uint32_t>;
#ifdef ARROW_FASTLANES_FUSED_FL_UNPACK
  for (size_t b = 0; b < num_blocks; ++b) {
    UnpackBlockFlToFileOrder<w, false>(packed + b * w * G::kLanes,
                                       out + b * kBlockSize);
  }
#else
  uint32_t scratch[kBlockSize];
  for (size_t b = 0; b < num_blocks; ++b) {
    UnpackBlock<uint32_t, w, false>(packed + b * w * G::kLanes, scratch);
    Transpose32x32(scratch, out + b * kBlockSize);
  }
#endif
}

template <typename T, uint32_t w>
void BM_Interleaved(benchmark::State& state, size_t out_bytes) {
  using G = BlockGeometry<T>;
  const size_t num_values = out_bytes / sizeof(T);
  const size_t num_blocks = num_values / kBlockSize;
  const size_t packed_elements = num_blocks * w * G::kLanes;
  auto packed = AllocateAligned<T>(packed_elements);
  RandomFill(packed.get(), packed_elements);
  auto out = AllocateAligned<T>(num_values);

  for (auto _ : state) {
    DecodePage<T, w>(packed.get(), out.get(), num_blocks);
    benchmark::DoNotOptimize(out[0]);
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(static_cast<int64_t>(out_bytes) * state.iterations());
  state.SetItemsProcessed(static_cast<int64_t>(num_values) * state.iterations());
}

template <uint32_t w>
void BM_InterleavedFileOrder(benchmark::State& state, size_t out_bytes) {
  using G = BlockGeometry<uint32_t>;
  const size_t num_values = out_bytes / sizeof(uint32_t);
  const size_t num_blocks = num_values / kBlockSize;
  const size_t packed_elements = num_blocks * w * G::kLanes;
  auto packed = AllocateAligned<uint32_t>(packed_elements);
  RandomFill(packed.get(), packed_elements);
  auto out = AllocateAligned<int32_t>(num_values);

  for (auto _ : state) {
    DecodePageFileOrder<w>(packed.get(), out.get(), num_blocks);
    benchmark::DoNotOptimize(out[0]);
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(static_cast<int64_t>(out_bytes) * state.iterations());
  state.SetItemsProcessed(static_cast<int64_t>(num_values) * state.iterations());
}

std::string Label(const char* decoder, uint32_t w, size_t element_bits,
                  const char* pinned, size_t pinned_kib) {
  return std::string("bitunpack/") + decoder + "/w=" + std::to_string(w) + "/u" +
         std::to_string(element_bits) + "/" + pinned + "=" +
         std::to_string(pinned_kib) + "KiB";
}

// A width needs one registration per decoder, and w has to be a constant for
// the interleaved kernel, so the widths are walked at compile time.
template <uint32_t w>
void RegisterWidth(size_t out_bytes, const char* pinned, size_t pinned_kib) {
  using T = NextWholeByte<w>;
  constexpr size_t kBits = sizeof(T) * 8;
  // A block is the decode unit, so a footprint below one block of this element
  // width has nothing to measure.
  if (out_bytes / sizeof(T) < kBlockSize) return;

  benchmark::RegisterBenchmark(Label("seq_simd", w, kBits, pinned, pinned_kib),
                               [out_bytes](benchmark::State& st) {
                                 BM_SequentialSimd<T, w>(st, out_bytes);
                               });
  benchmark::RegisterBenchmark(Label("seq_lib", w, kBits, pinned, pinned_kib),
                               [out_bytes](benchmark::State& st) {
                                 BM_Sequential<T, w>(st, out_bytes);
                               });
  benchmark::RegisterBenchmark(Label("seq_scal", w, kBits, pinned, pinned_kib),
                               [out_bytes](benchmark::State& st) {
                                 BM_SequentialScalar<T, w>(st, out_bytes);
                               });
  if constexpr (std::is_same_v<T, uint32_t>) {
    benchmark::RegisterBenchmark(
        Label("interleaved_file_order", w, kBits, pinned, pinned_kib),
        [out_bytes](benchmark::State& st) {
          BM_InterleavedFileOrder<w>(st, out_bytes);
        });
  }
  benchmark::RegisterBenchmark(Label("interleaved_fl_order", w, kBits, pinned,
                                     pinned_kib),
                               [out_bytes](benchmark::State& st) {
                                 BM_Interleaved<T, w>(st, out_bytes);
                               });
}

template <uint32_t... Ws>
void RegisterAllWidths(size_t out_bytes, std::integer_sequence<uint32_t, Ws...>) {
  (RegisterWidth<Ws + 1>(out_bytes, "out", out_bytes / 1024), ...);
}

// The width sweep holds the footprint inside L1 so that a row moves only with
// the bit width and the element it decodes into, and not with where the data
// had to come from. Both buffers count against L1, not just the decoded one:
// the packed source adds a further w bits per value, so a 16 KiB destination
// costs at most 32 KiB together and stays inside this core's 48 KiB of L1 data
// cache at every width. A 32 KiB destination would leave L1 for all but the
// four narrowest rows, and the sweep would then be moving the footprint it
// means to hold still.
constexpr size_t kWidthSweepOutBytes = 16 * 1024;

// The footprint sweep holds the bit width and walks the decoded footprint from
// inside L1 out to the largest page a decoder is ever handed. This core has
// 48 KiB of L1 data cache and 2 MiB of L2 to itself, so the ladder crosses
// both and ends just past L2. It stops at 4 MiB because that is already larger
// than a Parquet page, and the 480 MiB of L3 this socket shares is therefore
// out of reach: a page decoder never reads from memory on this machine, and a
// row that did would not describe one.
// Decoded destination size. The packed source adds a further w bits per value,
// so the footprint a row actually touches is larger by w/(8*sizeof(T)) and the
// cache level a row lands in depends on the width as well as the size listed
// here: 48 KiB of bytes at three bits totals 66 KiB and is already past this
// core's 48 KiB of L1. These sizes are therefore not labelled by level; the
// total and its level are computed per row, where both are known.
constexpr size_t kFootprintLadderKiB[] = {16,  32,  48,   64,   128,
                                          256, 512, 1024, 2048, 4096};

// One width per element size, each far from the degenerate cell where the bit
// width equals the output width and the decode is a vectorized copy.
void RegisterFootprintLadder() {
  for (size_t kib : kFootprintLadderKiB) {
    // The width sweep already holds these three widths at its own pin, under
    // the same name, so registering that size here would run it twice and
    // print two rows a reader keyed by name cannot tell apart.
    if (kib * 1024 == kWidthSweepOutBytes) continue;
    RegisterWidth<3>(kib * 1024, "out", kib);
    RegisterWidth<11>(kib * 1024, "out", kib);
    RegisterWidth<21>(kib * 1024, "out", kib);
  }
}

void RegisterBenchmarks() {
  RegisterAllWidths(kWidthSweepOutBytes, std::make_integer_sequence<uint32_t, 32>{});
  RegisterFootprintLadder();
}

}  // namespace

}  // namespace arrow::util::fastlanes

int main(int argc, char** argv) {
  arrow::util::fastlanes::RegisterBenchmarks();
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
