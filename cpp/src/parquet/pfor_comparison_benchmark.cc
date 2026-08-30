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

// Comparison benchmark: PFOR vs DeltaBitPack vs ZSTD vs RleBitPackHybrid
//                       vs ByteStreamSplit+ZSTD vs ByteStreamSplit+LZ4
//
// All throughput is reported as uncompressed_size / time (MB/s).
// Data generators mimic ClickBench and TPC-DS column distributions.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"

#include "arrow/util/bpacking_internal.h"
#include "arrow/util/bpacking_scalar_generated_internal.h"
#include "arrow/util/compression.h"
#include "arrow/util/fastlanes/fastlanes_for.h"
#include "arrow/util/fastlanes/lane_delta.h"
#include "arrow/util/logging.h"
#include "arrow/util/pfor/pfor_wrapper.h"
#include "arrow/util/rle_encoding_internal.h"

#include "parquet/encoding.h"
#include "parquet/platform.h"
#include "parquet/schema.h"
#include "parquet/types.h"

using ::arrow::Compression;
using ::arrow::util::Codec;

namespace parquet {
namespace {

// ============================================================================
// Data Generators — ClickBench-inspired
// ============================================================================

using Gen32 = std::vector<int32_t> (*)(int64_t);

std::vector<int32_t> GenClientIP(int64_t n) {
  std::vector<int32_t> v(n);
  std::mt19937 rng(101);
  std::uniform_int_distribution<uint32_t> dist(0x0A000000, 0xDFFFFFFF);
  for (auto& x : v) x = static_cast<int32_t>(dist(rng));
  return v;
}

std::vector<int32_t> GenUrlRegionID(int64_t n) {
  std::vector<int32_t> v(n);
  std::mt19937 rng(102);
  // Zipf-like over ~1000 values
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  for (auto& x : v) {
    double u = uni(rng);
    x = static_cast<int32_t>(std::pow(u, 2.0) * 1000) + 1;
  }
  return v;
}

std::vector<int32_t> GenCounterID(int64_t n) {
  std::vector<int32_t> v(n);
  std::mt19937 rng(103);
  std::uniform_int_distribution<int32_t> jitter(0, 3);
  int32_t counter = 100000;
  for (auto& x : v) {
    counter += 1 + jitter(rng);
    x = counter;
  }
  return v;
}

std::vector<int32_t> GenEventDate(int64_t n) {
  std::vector<int32_t> v(n);
  std::mt19937 rng(104);
  const int32_t dates[] = {19691, 19692, 19693, 19694, 19695};
  std::uniform_int_distribution<int> idx(0, 4);
  for (auto& x : v) x = dates[idx(rng)];
  return v;
}

std::vector<int32_t> GenEventTime(int64_t n) {
  std::vector<int32_t> v(n);
  std::mt19937 rng(105);
  const int32_t base = 1704067200;  // 2024-01-01
  std::uniform_int_distribution<int32_t> offset(0, 86399);
  for (auto& x : v) x = base + offset(rng);
  return v;
}

std::vector<int32_t> GenGoodEvent(int64_t n) {
  std::vector<int32_t> v(n);
  std::mt19937 rng(106);
  std::uniform_int_distribution<int> dist(0, 99);
  for (auto& x : v) x = (dist(rng) < 95) ? 1 : 0;
  return v;
}

std::vector<int32_t> GenHID(int64_t n) {
  std::vector<int32_t> v(n);
  std::mt19937 rng(107);
  std::uniform_int_distribution<int32_t> dist(std::numeric_limits<int32_t>::min(),
                                              std::numeric_limits<int32_t>::max());
  for (auto& x : v) x = dist(rng);
  return v;
}

std::vector<int32_t> GenHitColor(int64_t n) {
  std::vector<int32_t> v(n);
  std::mt19937 rng(108);
  const int32_t colors[] = {1, 2, 3, 4, 5};
  std::uniform_int_distribution<int> idx(0, 4);
  for (auto& x : v) x = colors[idx(rng)];
  return v;
}

std::vector<int32_t> GenIPNetworkID(int64_t n) {
  std::vector<int32_t> v(n);
  std::mt19937 rng(109);
  std::uniform_int_distribution<int32_t> dist(1, 10000);
  for (auto& x : v) x = dist(rng);
  return v;
}

std::vector<int32_t> GenJavaEnable(int64_t n) {
  std::vector<int32_t> v(n);
  std::mt19937 rng(110);
  std::uniform_int_distribution<int> dist(0, 99);
  for (auto& x : v) x = (dist(rng) < 85) ? 1 : 0;
  return v;
}

std::vector<int32_t> GenOS(int64_t n) {
  std::vector<int32_t> v(n);
  std::mt19937 rng(111);
  std::uniform_int_distribution<int32_t> dist(1, 20);
  for (auto& x : v) x = dist(rng);
  return v;
}

std::vector<int32_t> GenResolution(int64_t n) {
  std::vector<int32_t> v(n);
  std::mt19937 rng(112);
  const int32_t resolutions[] = {360,  480,  600,  720,  768,  800,  900,
                                 1024, 1050, 1080, 1200, 1440, 1600, 2160};
  std::uniform_int_distribution<int> idx(0, 13);
  for (auto& x : v) x = resolutions[idx(rng)];
  return v;
}

std::vector<int32_t> GenTrafficSourceID(int64_t n) {
  std::vector<int32_t> v(n);
  std::mt19937 rng(113);
  std::uniform_int_distribution<int32_t> dist(0, 10);
  for (auto& x : v) x = dist(rng);
  return v;
}

std::vector<int32_t> GenUserAgent(int64_t n) {
  std::vector<int32_t> v(n);
  std::mt19937 rng(114);
  // Zipf-like over ~100 user agents
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  for (auto& x : v) {
    double u = uni(rng);
    x = static_cast<int32_t>(std::pow(u, 1.5) * 100) + 1;
  }
  return v;
}

// ============================================================================
// Data Generators — TPC-DS (4 most queried columns from store_sales)
// ============================================================================

std::vector<int32_t> GenTpcdsSoldDateSk(int64_t n) {
  std::vector<int32_t> v(n);
  const int32_t kBase = 2450815;
  std::mt19937 rng(201);
  std::uniform_int_distribution<int32_t> dist(0, 1820);
  for (auto& x : v) x = kBase + dist(rng);
  return v;
}

std::vector<int32_t> GenTpcdsStoreSk(int64_t n) {
  std::vector<int32_t> v(n);
  std::mt19937 rng(202);
  std::uniform_int_distribution<int32_t> dist(1, 1000);
  for (auto& x : v) x = dist(rng);
  return v;
}

std::vector<int32_t> GenTpcdsItemSk(int64_t n) {
  std::vector<int32_t> v(n);
  const int32_t kMax = 100000;
  std::mt19937 rng(203);
  std::exponential_distribution<double> exp_dist(0.00005);
  for (auto& x : v) {
    int32_t val = static_cast<int32_t>(exp_dist(rng));
    x = std::min(val + 1, kMax);
  }
  return v;
}

std::vector<int32_t> GenTpcdsQuantity(int64_t n) {
  std::vector<int32_t> v(n);
  std::mt19937 rng(204);
  std::uniform_int_distribution<int32_t> small_dist(1, 10);
  std::uniform_int_distribution<int32_t> large_dist(11, 100);
  std::uniform_int_distribution<int> chance(0, 99);
  for (auto& x : v) {
    x = (chance(rng) < 90) ? small_dist(rng) : large_dist(rng);
  }
  return v;
}

// --- TPC-H (lineitem) top-queried numeric columns (Q1/Q3/Q5/Q6) ------------
// l_quantity: integer [1, 50], uniform. Small range, min 1.
std::vector<int32_t> GenTpchLQuantity(int64_t n) {
  std::vector<int32_t> v(n);
  std::mt19937 rng(301);
  std::uniform_int_distribution<int32_t> dist(1, 50);
  for (auto& x : v) x = dist(rng);
  return v;
}

// l_extendedprice (cents): l_quantity * p_retailprice. p_retailprice spans
// ~$900.00..$2099.00, so cents in [90000, 10495000]. Wide range, nonzero min.
std::vector<int32_t> GenTpchLExtendedPrice(int64_t n) {
  std::vector<int32_t> v(n);
  std::mt19937 rng(302);
  std::uniform_int_distribution<int32_t> qty(1, 50);
  std::uniform_int_distribution<int32_t> retail_cents(90000, 209900);
  for (auto& x : v) x = qty(rng) * retail_cents(rng);
  return v;
}

// l_discount (x100): integer [0, 10] i.e. 0.00..0.10. Genuinely includes 0
// (0% discount is a real value), so this one legitimately starts at 0.
std::vector<int32_t> GenTpchLDiscount(int64_t n) {
  std::vector<int32_t> v(n);
  std::mt19937 rng(303);
  std::uniform_int_distribution<int32_t> dist(0, 10);
  for (auto& x : v) x = dist(rng);
  return v;
}

// l_shipdate (days since 1970-01-01): 1992-01-01..1998-12 span. Base 8036,
// range ~7 years. Large nonzero min -> exercises frame-of-reference.
std::vector<int32_t> GenTpchLShipDate(int64_t n) {
  std::vector<int32_t> v(n);
  const int32_t kBase = 8036;  // days since epoch for 1992-01-01
  std::mt19937 rng(304);
  std::uniform_int_distribution<int32_t> dist(0, 2557);
  for (auto& x : v) x = kBase + dist(rng);
  return v;
}

// --- TPC-DS (store_sales / date_dim) further top-queried numeric columns ---
// ss_customer_sk: surrogate key, uniform [1, 2,000,000]. Big range, min 1.
std::vector<int32_t> GenTpcdsCustomerSk(int64_t n) {
  std::vector<int32_t> v(n);
  std::mt19937 rng(311);
  std::uniform_int_distribution<int32_t> dist(1, 2000000);
  for (auto& x : v) x = dist(rng);
  return v;
}

// ss_ext_sales_price (cents): skewed price, exponential mean ~$50, floored at
// $1.00 (a sale has a nonzero price), capped at $20,000. Long tail -> patches.
std::vector<int32_t> GenTpcdsExtSalesPrice(int64_t n) {
  std::vector<int32_t> v(n);
  const int32_t kMin = 100, kMax = 2000000;
  std::mt19937 rng(312);
  std::exponential_distribution<double> exp_dist(1.0 / 5000.0);
  for (auto& x : v) {
    int32_t val = kMin + static_cast<int32_t>(exp_dist(rng));
    x = std::min(val, kMax);
  }
  return v;
}

// ss_net_profit (cents): usually a small profit, sometimes a loss -> negative
// values, so the frame of reference is negative (not zero).
std::vector<int32_t> GenTpcdsNetProfit(int64_t n) {
  std::vector<int32_t> v(n);
  std::mt19937 rng(313);
  std::uniform_int_distribution<int32_t> dist(-10000, 300000);
  for (auto& x : v) x = dist(rng);
  return v;
}

// d_year: queried date_dim year range [1998, 2003]. Low cardinality, min 1998.
std::vector<int32_t> GenTpcdsDYear(int64_t n) {
  std::vector<int32_t> v(n);
  std::mt19937 rng(314);
  std::uniform_int_distribution<int32_t> dist(1998, 2003);
  for (auto& x : v) x = dist(rng);
  return v;
}

// --- NYC yellow-taxi trip numeric columns ----------------------------------
// pickup timestamp (unix seconds): 2015-01, base 1,420,070,400 + ~31 days.
// Very large min -> frame-of-reference is essential.
std::vector<int32_t> GenTaxiPickupUnixTime(int64_t n) {
  std::vector<int32_t> v(n);
  const int32_t kBase = 1420070400;  // 2015-01-01 UTC
  std::mt19937 rng(321);
  std::uniform_int_distribution<int32_t> dist(0, 2678400);  // ~31 days
  for (auto& x : v) x = kBase + dist(rng);
  return v;
}

// trip_distance (x100 miles): exponential mean ~1.8 mi, floored at 0.10 mi
// (no zero-distance trips), capped at 100 mi. Long tail -> patches.
std::vector<int32_t> GenTaxiTripDistanceX100(int64_t n) {
  std::vector<int32_t> v(n);
  const int32_t kMin = 10, kMax = 10000;
  std::mt19937 rng(322);
  std::exponential_distribution<double> exp_dist(1.0 / 180.0);
  for (auto& x : v) {
    int32_t val = kMin + static_cast<int32_t>(exp_dist(rng));
    x = std::min(val, kMax);
  }
  return v;
}

// fare_amount (cents): $2.50 base + exponential mean ~$10, capped at $150.
// Nonzero floor at the base fare, skewed with a long tail.
std::vector<int32_t> GenTaxiFareCents(int64_t n) {
  std::vector<int32_t> v(n);
  const int32_t kBase = 250, kMax = 15000;
  std::mt19937 rng(323);
  std::exponential_distribution<double> exp_dist(1.0 / 1000.0);
  for (auto& x : v) {
    int32_t val = kBase + static_cast<int32_t>(exp_dist(rng));
    x = std::min(val, kMax);
  }
  return v;
}

// ============================================================================
// Data Generators — sorted and near-sorted
//
// The generators above draw independently around a base, which is the case
// delta encoding is not for: subtracting two i.i.d. values spans the whole
// range whichever two you pick, so the distance between them does not matter
// and every delta variant lands within a bit or two of frame of reference.
// Comparing delta schemes needs columns whose value depends on the previous
// one. These cover the shapes that occur in practice: a clustered timestamp
// column, a sorted key with duplicates, an exact counter, and a column that
// is sorted apart from a few late arrivals.
// ============================================================================

// Event timestamps in seconds, arriving a few seconds apart. The common case
// for a table clustered or sorted on time.
std::vector<int32_t> GenSortedUnixTime(int64_t n) {
  std::vector<int32_t> v(n);
  std::mt19937 rng(401);
  std::uniform_int_distribution<int32_t> gap(0, 7);
  int32_t t = 1700000000;
  for (auto& x : v) {
    t += gap(rng);
    x = t;
  }
  return v;
}

// A sorted surrogate key with runs of duplicates, as produced by a join key or
// a dictionary-sorted column: most steps are 0, some are 1.
std::vector<int32_t> GenSortedKeyDups(int64_t n) {
  std::vector<int32_t> v(n);
  std::mt19937 rng(402);
  std::uniform_int_distribution<int32_t> step(0, 1);
  int32_t k = 5000000;
  for (auto& x : v) {
    k += step(rng);
    x = k;
  }
  return v;
}

// An exact +1 row id. The best case for any delta scheme, and the case where
// stride-1 and lane-parallel delta differ most.
std::vector<int32_t> GenMonotoneRowId(int64_t n) {
  std::vector<int32_t> v(n);
  int32_t k = 1;
  for (auto& x : v) x = k++;
  return v;
}

// Sorted except that 2% of rows arrive late, so a few deltas are large and
// negative. Tests whether one bit width per block survives outliers.
std::vector<int32_t> GenNearSortedUnixTime(int64_t n) {
  std::vector<int32_t> v = GenSortedUnixTime(n);
  std::mt19937 rng(403);
  std::uniform_int_distribution<int> pick(0, 49);
  std::uniform_int_distribution<int32_t> late(1, 3600);
  for (auto& x : v) {
    if (pick(rng) == 0) x -= late(rng);
  }
  return v;
}

// ============================================================================
// Helpers
// ============================================================================

static int32_t ComputeBitWidth(const std::vector<int32_t>& values) {
  uint32_t max_val = 0;
  for (int32_t v : values) {
    max_val = std::max(max_val, static_cast<uint32_t>(v));
  }
  if (max_val == 0) return 1;
  return static_cast<int32_t>(32 - __builtin_clz(max_val));
}

static std::shared_ptr<ColumnDescriptor> MakeInt32Descriptor() {
  auto node =
      schema::PrimitiveNode::Make("col", Repetition::REQUIRED, Type::INT32);
  return std::make_shared<ColumnDescriptor>(node, /*max_def_level=*/0,
                                            /*max_rep_level=*/0);
}

// ============================================================================
// PFOR Encode/Decode
// ============================================================================

static void BM_PforEncode(benchmark::State& state, Gen32 gen) {
  const int64_t num_values = state.range(0);
  auto values = gen(num_values);
  const int64_t uncompressed_size = num_values * sizeof(int32_t);

  int64_t max_size =
      ::arrow::util::pfor::PforWrapper<int32_t>::GetMaxCompressedSize(
          static_cast<int32_t>(num_values));
  std::vector<uint8_t> compressed(max_size);

  // Compute comp_size once for the counter
  int64_t comp_size = max_size;
  ::arrow::util::pfor::PforWrapper<int32_t>::Encode(
      values.data(), static_cast<int32_t>(num_values), compressed.data(), &comp_size);

  for (auto _ : state) {
    int64_t sz = max_size;
    ::arrow::util::pfor::PforWrapper<int32_t>::Encode(
        values.data(), static_cast<int32_t>(num_values), compressed.data(), &sz);
    benchmark::DoNotOptimize(sz);
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * uncompressed_size);
  state.SetItemsProcessed(state.iterations() * num_values);
  state.counters["compression_ratio"] =
      static_cast<double>(uncompressed_size) / static_cast<double>(comp_size);
}

static void BM_PforDecode(benchmark::State& state, Gen32 gen) {
  const int64_t num_values = state.range(0);
  auto values = gen(num_values);
  const int64_t uncompressed_size = num_values * sizeof(int32_t);

  int64_t max_size =
      ::arrow::util::pfor::PforWrapper<int32_t>::GetMaxCompressedSize(
          static_cast<int32_t>(num_values));
  std::vector<uint8_t> compressed(max_size);
  int64_t comp_size = max_size;
  ::arrow::util::pfor::PforWrapper<int32_t>::Encode(
      values.data(), static_cast<int32_t>(num_values), compressed.data(),
      &comp_size);

  std::vector<int32_t> decoded(num_values);
  for (auto _ : state) {
    auto status = ::arrow::util::pfor::PforWrapper<int32_t>::Decode(
        decoded.data(), static_cast<int32_t>(num_values), compressed.data(),
        comp_size);
    ARROW_CHECK_OK(status);
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * uncompressed_size);
  state.SetItemsProcessed(state.iterations() * num_values);
  state.counters["compression_ratio"] =
      static_cast<double>(uncompressed_size) / static_cast<double>(comp_size);
}

// ============================================================================
// PFOR with FastLanes-mode bit packing (per-vector flag in PFOR header)
// ============================================================================
//
// Same PFOR pipeline (FOR + exceptions) but the bit-packing payload uses
// FastLanes lane-interleaved layout instead of the legacy sequential bit
// stream. Output is still flat (PFOR contract preserved); the decoder
// scatters via FL_ORDER inside DecodeVector.

static void BM_PforFastLanesEncode(benchmark::State& state, Gen32 gen) {
  const int64_t num_values = state.range(0);
  auto values = gen(num_values);
  const int64_t uncompressed_size = num_values * sizeof(int32_t);

  int64_t max_size =
      ::arrow::util::pfor::PforWrapper<int32_t>::GetMaxCompressedSize(
          static_cast<int32_t>(num_values));
  std::vector<uint8_t> compressed(max_size);

  int64_t comp_size = max_size;
  ::arrow::util::pfor::PforWrapper<int32_t>::Encode(
      values.data(), static_cast<int32_t>(num_values), compressed.data(), &comp_size,
      ::arrow::util::pfor::PackingMode::FastLanes);

  for (auto _ : state) {
    int64_t sz = max_size;
    ::arrow::util::pfor::PforWrapper<int32_t>::Encode(
        values.data(), static_cast<int32_t>(num_values), compressed.data(), &sz,
        ::arrow::util::pfor::PackingMode::FastLanes);
    benchmark::DoNotOptimize(sz);
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * uncompressed_size);
  state.SetItemsProcessed(state.iterations() * num_values);
  state.counters["compression_ratio"] =
      static_cast<double>(uncompressed_size) / static_cast<double>(comp_size);
}

static void BM_PforFastLanesDecode(benchmark::State& state, Gen32 gen) {
  const int64_t num_values = state.range(0);
  auto values = gen(num_values);
  const int64_t uncompressed_size = num_values * sizeof(int32_t);

  int64_t max_size =
      ::arrow::util::pfor::PforWrapper<int32_t>::GetMaxCompressedSize(
          static_cast<int32_t>(num_values));
  std::vector<uint8_t> compressed(max_size);
  int64_t comp_size = max_size;
  ::arrow::util::pfor::PforWrapper<int32_t>::Encode(
      values.data(), static_cast<int32_t>(num_values), compressed.data(),
      &comp_size, ::arrow::util::pfor::PackingMode::FastLanes);

  std::vector<int32_t> decoded(num_values);
  for (auto _ : state) {
    auto status = ::arrow::util::pfor::PforWrapper<int32_t>::Decode(
        decoded.data(), static_cast<int32_t>(num_values), compressed.data(),
        comp_size);
    ARROW_CHECK_OK(status);
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * uncompressed_size);
  state.SetItemsProcessed(state.iterations() * num_values);
  state.counters["compression_ratio"] =
      static_cast<double>(uncompressed_size) / static_cast<double>(comp_size);
}

// PFOR with FastLanes bit-packing, decoded in TRANSPOSED order: skips the
// per-vector FL_ORDER gather, output is in FastLanes stream order. The
// downstream consumer must be permutation-aware. Apples-to-apples this
// against BM_PforFastLanesDecode (flat-order, with the scatter cost) to see
// how much of the gap to pfor+bitpack closes.
static void BM_PforFastLanesDecodeTransposed(benchmark::State& state, Gen32 gen) {
  const int64_t num_values = state.range(0);
  auto values = gen(num_values);
  const int64_t uncompressed_size = num_values * sizeof(int32_t);

  int64_t max_size =
      ::arrow::util::pfor::PforWrapper<int32_t>::GetMaxCompressedSize(
          static_cast<int32_t>(num_values));
  std::vector<uint8_t> compressed(max_size);
  int64_t comp_size = max_size;
  ::arrow::util::pfor::PforWrapper<int32_t>::Encode(
      values.data(), static_cast<int32_t>(num_values), compressed.data(),
      &comp_size, ::arrow::util::pfor::PackingMode::FastLanes);

  std::vector<int32_t> decoded(num_values);
  for (auto _ : state) {
    auto status = ::arrow::util::pfor::PforWrapper<int32_t>::Decode(
        decoded.data(), static_cast<int32_t>(num_values), compressed.data(),
        comp_size, ::arrow::util::pfor::OutputOrder::Transposed);
    ARROW_CHECK_OK(status);
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * uncompressed_size);
  state.SetItemsProcessed(state.iterations() * num_values);
  state.counters["compression_ratio"] =
      static_cast<double>(uncompressed_size) / static_cast<double>(comp_size);
}

// PFOR with FastLanes interleaved bit-packing but WITHOUT the FL_ORDER reorder
// (PackingMode::FastLanesOrdered). Decodes to flat (original) order at full
// unpack speed with no gather — the point being that for FOR/bit-packing the
// reorder (and its flat-decode penalty) is unnecessary. Same compression ratio.
static void BM_PforFastLanesOrderedEncode(benchmark::State& state, Gen32 gen) {
  const int64_t num_values = state.range(0);
  auto values = gen(num_values);
  const int64_t uncompressed_size = num_values * sizeof(int32_t);

  int64_t max_size =
      ::arrow::util::pfor::PforWrapper<int32_t>::GetMaxCompressedSize(
          static_cast<int32_t>(num_values));
  std::vector<uint8_t> compressed(max_size);
  int64_t comp_size = max_size;

  for (auto _ : state) {
    int64_t sz = max_size;
    ::arrow::util::pfor::PforWrapper<int32_t>::Encode(
        values.data(), static_cast<int32_t>(num_values), compressed.data(), &sz,
        ::arrow::util::pfor::PackingMode::FastLanesOrdered);
    benchmark::DoNotOptimize(sz);
    comp_size = sz;
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * uncompressed_size);
  state.SetItemsProcessed(state.iterations() * num_values);
  state.counters["compression_ratio"] =
      static_cast<double>(uncompressed_size) / static_cast<double>(comp_size);
}

static void BM_PforFastLanesOrderedDecode(benchmark::State& state, Gen32 gen) {
  const int64_t num_values = state.range(0);
  auto values = gen(num_values);
  const int64_t uncompressed_size = num_values * sizeof(int32_t);

  int64_t max_size =
      ::arrow::util::pfor::PforWrapper<int32_t>::GetMaxCompressedSize(
          static_cast<int32_t>(num_values));
  std::vector<uint8_t> compressed(max_size);
  int64_t comp_size = max_size;
  ::arrow::util::pfor::PforWrapper<int32_t>::Encode(
      values.data(), static_cast<int32_t>(num_values), compressed.data(),
      &comp_size, ::arrow::util::pfor::PackingMode::FastLanesOrdered);

  std::vector<int32_t> decoded(num_values);
  for (auto _ : state) {
    // Default OutputOrder::Flat — ordered mode always returns original order.
    auto status = ::arrow::util::pfor::PforWrapper<int32_t>::Decode(
        decoded.data(), static_cast<int32_t>(num_values), compressed.data(),
        comp_size);
    ARROW_CHECK_OK(status);
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * uncompressed_size);
  state.SetItemsProcessed(state.iterations() * num_values);
  state.counters["compression_ratio"] =
      static_cast<double>(uncompressed_size) / static_cast<double>(comp_size);
}

// ============================================================================
// DeltaBitPack Encode/Decode
// ============================================================================

static void BM_DeltaBitPackEncode(benchmark::State& state, Gen32 gen) {
  const int64_t num_values = state.range(0);
  auto values = gen(num_values);
  const int64_t uncompressed_size = num_values * sizeof(int32_t);

  auto encoder = MakeTypedEncoder<Int32Type>(Encoding::DELTA_BINARY_PACKED);

  // Compute comp_size once for the counter
  encoder->Put(values.data(), static_cast<int>(num_values));
  auto pre_buf = encoder->FlushValues();
  int64_t comp_size = pre_buf->size();

  for (auto _ : state) {
    encoder->Put(values.data(), static_cast<int>(num_values));
    auto buf = encoder->FlushValues();
    benchmark::DoNotOptimize(buf);
  }

  state.SetBytesProcessed(state.iterations() * uncompressed_size);
  state.SetItemsProcessed(state.iterations() * num_values);
  state.counters["compression_ratio"] =
      static_cast<double>(uncompressed_size) / static_cast<double>(comp_size);
}

static void BM_DeltaBitPackDecode(benchmark::State& state, Gen32 gen) {
  const int64_t num_values = state.range(0);
  auto values = gen(num_values);
  const int64_t uncompressed_size = num_values * sizeof(int32_t);

  auto encoder = MakeTypedEncoder<Int32Type>(Encoding::DELTA_BINARY_PACKED);
  encoder->Put(values.data(), static_cast<int>(num_values));
  auto buf = encoder->FlushValues();
  int64_t comp_size = buf->size();

  std::vector<int32_t> decoded(num_values);
  auto decoder = MakeTypedDecoder<Int32Type>(Encoding::DELTA_BINARY_PACKED);

  for (auto _ : state) {
    decoder->SetData(static_cast<int>(num_values), buf->data(),
                     static_cast<int>(buf->size()));
    decoder->Decode(decoded.data(), static_cast<int>(num_values));
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * uncompressed_size);
  state.SetItemsProcessed(state.iterations() * num_values);
  state.counters["compression_ratio"] =
      static_cast<double>(uncompressed_size) / static_cast<double>(comp_size);
}

// ============================================================================
// DELTA_BINARY_PACKED with 1024-value blocks
//
// values_per_block and mini_blocks_per_block are stream header fields, not
// constants: the format requires only that the block be a multiple of 128 and
// the mini-block a multiple of 32. Arrow's encoder takes both as constructor
// arguments but defaults to 128/4 and does not expose them through
// MakeTypedEncoder, so this hand-builds the stream to find out what the
// existing format can do with one bit width per 1024 values. Only correct for
// a constant stride, which is the case being checked.
// ============================================================================

static void PutVlq(std::vector<uint8_t>* out, uint32_t v) {
  while (v >= 0x80) {
    out->push_back(static_cast<uint8_t>((v & 0x7F) | 0x80));
    v >>= 7;
  }
  out->push_back(static_cast<uint8_t>(v));
}

static void PutZigZagVlq(std::vector<uint8_t>* out, int32_t v) {
  PutVlq(out, (static_cast<uint32_t>(v) << 1) ^ static_cast<uint32_t>(v >> 31));
}

static void BM_DeltaBitPackBigBlockDecode(benchmark::State& state, Gen32 gen) {
  const int64_t num_values = state.range(0);
  auto values = gen(num_values);
  const int64_t uncompressed_size = num_values * sizeof(int32_t);

  const int32_t stride = values[1] - values[0];
  for (int64_t i = 1; i < num_values; ++i) {
    ARROW_CHECK(values[i] - values[i - 1] == stride) << "stride is not constant";
  }

  const uint32_t kValuesPerBlock = 1024;
  std::vector<uint8_t> buf;
  PutVlq(&buf, kValuesPerBlock);
  PutVlq(&buf, 1);  // one mini-block, so one bit width per 1024 values
  PutVlq(&buf, static_cast<uint32_t>(num_values));
  PutZigZagVlq(&buf, values[0]);
  const int64_t nblocks =
      (num_values - 1 + kValuesPerBlock - 1) / kValuesPerBlock;
  for (int64_t b = 0; b < nblocks; ++b) {
    PutZigZagVlq(&buf, stride);  // min delta absorbs the whole step
    buf.push_back(0);            // bit width 0: nothing to pack
  }
  const int64_t comp_size = static_cast<int64_t>(buf.size());

  std::vector<int32_t> decoded(num_values);
  auto decoder = MakeTypedDecoder<Int32Type>(Encoding::DELTA_BINARY_PACKED);
  decoder->SetData(static_cast<int>(num_values), buf.data(),
                   static_cast<int>(buf.size()));
  decoder->Decode(decoded.data(), static_cast<int>(num_values));
  ARROW_CHECK(decoded == values) << "hand-built 1024-block stream did not round trip";

  for (auto _ : state) {
    decoder->SetData(static_cast<int>(num_values), buf.data(),
                     static_cast<int>(buf.size()));
    decoder->Decode(decoded.data(), static_cast<int>(num_values));
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * uncompressed_size);
  state.SetItemsProcessed(state.iterations() * num_values);
  state.counters["compression_ratio"] =
      static_cast<double>(uncompressed_size) / static_cast<double>(comp_size);
  state.counters["bytes"] = static_cast<double>(comp_size);
}

// ============================================================================
// Lane-parallel delta Encode/Decode
//
// Same idea as DELTA_BINARY_PACKED, but the delta runs down a bit-packing lane
// instead of along the file order, so the prefix sum becomes 32 independent
// chains of length 32 rather than one chain of length 1024.
// ============================================================================

namespace {

using ::arrow::util::fastlanes::kBlockSize;
using ::arrow::util::fastlanes::kLanes;
using ::arrow::util::fastlanes::kRowsPerBlock;
using ::arrow::util::fastlanes::LaneDeltaBitsFor;
using ::arrow::util::fastlanes::LaneDeltaDecode;
using ::arrow::util::fastlanes::LaneDeltaEncode;
using ::arrow::util::fastlanes::LaneDeltaMaxEncodedSize;
using ::arrow::util::fastlanes::LaneDeltaOrder;

// Encode, and check the round trip while doing so: a codec that decoded to the
// wrong values would still report a ratio and a throughput, and neither figure
// would mean anything.
template <LaneDeltaOrder kOrder>
int64_t LaneDeltaPrepare(const std::vector<int32_t>& values,
                         std::vector<uint8_t>* compressed) {
  compressed->assign(LaneDeltaMaxEncodedSize(values.size()), 0);
  const size_t comp_size =
      LaneDeltaEncode<kOrder>(values.data(), values.size(), compressed->data());
  ARROW_CHECK(comp_size > 0) << "lane delta: block span exceeded 32 bits";

  std::vector<int32_t> check(values.size());
  LaneDeltaDecode<kOrder, true>(compressed->data(), values.size(), check.data());
  ARROW_CHECK(check == values) << "lane delta round trip mismatch";
  return static_cast<int64_t>(comp_size);
}

template <LaneDeltaOrder kOrder>
void LaneDeltaEncodeBench(benchmark::State& state, Gen32 gen) {
  int64_t num_values = state.range(0);
  num_values -= num_values % static_cast<int64_t>(kBlockSize);
  auto values = gen(num_values);
  const int64_t uncompressed_size = num_values * sizeof(int32_t);

  std::vector<uint8_t> compressed;
  const int64_t comp_size = LaneDeltaPrepare<kOrder>(values, &compressed);

  for (auto _ : state) {
    size_t sz = LaneDeltaEncode<kOrder>(values.data(), num_values, compressed.data());
    benchmark::DoNotOptimize(sz);
  }

  state.SetBytesProcessed(state.iterations() * uncompressed_size);
  state.SetItemsProcessed(state.iterations() * num_values);
  state.counters["compression_ratio"] =
      static_cast<double>(uncompressed_size) / static_cast<double>(comp_size);
}

template <LaneDeltaOrder kOrder, bool kRepair>
void LaneDeltaDecodeBench(benchmark::State& state, Gen32 gen) {
  int64_t num_values = state.range(0);
  num_values -= num_values % static_cast<int64_t>(kBlockSize);
  auto values = gen(num_values);
  const int64_t uncompressed_size = num_values * sizeof(int32_t);

  std::vector<uint8_t> compressed;
  const int64_t comp_size = LaneDeltaPrepare<kOrder>(values, &compressed);

  std::vector<int32_t> decoded(num_values);
  for (auto _ : state) {
    LaneDeltaDecode<kOrder, kRepair>(compressed.data(), num_values, decoded.data());
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * uncompressed_size);
  state.SetItemsProcessed(state.iterations() * num_values);
  state.counters["compression_ratio"] =
      static_cast<double>(uncompressed_size) / static_cast<double>(comp_size);
}

// ============================================================================
// Width-granularity study
//
// Size accounting only, no packing: the question is whether a bit width finer
// than one per 1024-value block is worth the machinery, and that is answerable
// from the deltas alone. Reports bits per value under three schemes so they can
// be compared on the same column:
//
//   per_block    one width and one min for the whole block. What the codec does.
//   per_row      one width per row of 32 lanes, one min per block.
//   per_row_min  one width and one min per row.
//
// Mins are counted as zigzag varints in the per-row schemes, since 32
// fixed-width mins per block would cost a bit per value on their own and no
// real design would pay that.
// ============================================================================

size_t ZigZagVarintSize(int64_t v) {
  uint64_t z = (static_cast<uint64_t>(v) << 1) ^ static_cast<uint64_t>(v >> 63);
  size_t bytes = 1;
  while (z >= 0x80) {
    ++bytes;
    z >>= 7;
  }
  return bytes;
}

void BM_LaneDeltaWidthStudy(benchmark::State& state, Gen32 gen) {
  int64_t num_values = state.range(0);
  num_values -= num_values % static_cast<int64_t>(kBlockSize);
  auto values = gen(num_values);
  const int64_t nblocks = num_values / static_cast<int64_t>(kBlockSize);

  // Seeds, once per column, shared by every scheme.
  size_t bytes_a = kLanes * sizeof(uint32_t);
  size_t bytes_b = bytes_a;
  size_t bytes_c = bytes_a;

  std::vector<int64_t> deltas(kBlockSize);
  int32_t carry[kLanes];

  for (int64_t b = 0; b < nblocks; ++b) {
    const int32_t* blk = values.data() + b * static_cast<int64_t>(kBlockSize);
    if (b == 0) {
      for (size_t lane = 0; lane < kLanes; ++lane) carry[lane] = blk[lane];
    }
    for (size_t lane = 0; lane < kLanes; ++lane) {
      int64_t prev = carry[lane];
      for (size_t row = 0; row < kRowsPerBlock; ++row) {
        const int64_t cur = blk[row * kLanes + lane];
        deltas[row * kLanes + lane] = cur - prev;
        prev = cur;
      }
      carry[lane] = blk[(kRowsPerBlock - 1) * kLanes + lane];
    }

    int64_t block_min = std::numeric_limits<int64_t>::max();
    for (size_t t = 0; t < kBlockSize; ++t) block_min = std::min(block_min, deltas[t]);

    uint64_t block_span = 0;
    for (size_t t = 0; t < kBlockSize; ++t) {
      block_span = std::max(block_span, static_cast<uint64_t>(deltas[t] - block_min));
    }
    const uint32_t wa = LaneDeltaBitsFor(block_span);
    bytes_a += 1 + sizeof(int32_t) + wa * kLanes * sizeof(uint32_t);

    size_t sum_b = 0, sum_c = 0, min_bytes_c = 0;
    for (size_t row = 0; row < kRowsPerBlock; ++row) {
      const int64_t* r = deltas.data() + row * kLanes;
      int64_t row_min = std::numeric_limits<int64_t>::max();
      int64_t row_max = std::numeric_limits<int64_t>::min();
      for (size_t lane = 0; lane < kLanes; ++lane) {
        row_min = std::min(row_min, r[lane]);
        row_max = std::max(row_max, r[lane]);
      }
      sum_b += LaneDeltaBitsFor(static_cast<uint64_t>(row_max - block_min));
      sum_c += LaneDeltaBitsFor(static_cast<uint64_t>(row_max - row_min));
      min_bytes_c += ZigZagVarintSize(row_min);
    }
    bytes_b += kRowsPerBlock + ZigZagVarintSize(block_min) + sum_b * sizeof(uint32_t);
    bytes_c += kRowsPerBlock + min_bytes_c + sum_c * sizeof(uint32_t);
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(bytes_a);
  }

  const double n = static_cast<double>(num_values);
  state.counters["per_block_bits"] = static_cast<double>(bytes_a) * 8.0 / n;
  state.counters["per_row_bits"] = static_cast<double>(bytes_b) * 8.0 / n;
  state.counters["per_row_min_bits"] = static_cast<double>(bytes_c) * 8.0 / n;
}

}  // namespace

// Interleaved order: container order is file order, so no permutation either way.
static void BM_LaneDeltaEncode(benchmark::State& state, Gen32 gen) {
  LaneDeltaEncodeBench<LaneDeltaOrder::kInterleaved>(state, gen);
}

static void BM_LaneDeltaDecode(benchmark::State& state, Gen32 gen) {
  LaneDeltaDecodeBench<LaneDeltaOrder::kInterleaved, true>(state, gen);
}

// FL_ORDER: smaller in-lane steps, but one width per block must cover the
// sub-block jumps, and decode owes a scatter back to file order.
static void BM_LaneDeltaFlOrderEncode(benchmark::State& state, Gen32 gen) {
  LaneDeltaEncodeBench<LaneDeltaOrder::kFlOrder>(state, gen);
}

static void BM_LaneDeltaFlOrderDecode(benchmark::State& state, Gen32 gen) {
  LaneDeltaDecodeBench<LaneDeltaOrder::kFlOrder, true>(state, gen);
}

// FL_ORDER stopping in transposed order: separates the container's cost from
// the permutation's, and is not a usable decoder on its own.
static void BM_LaneDeltaFlOrderDecodeTransposed(benchmark::State& state, Gen32 gen) {
  LaneDeltaDecodeBench<LaneDeltaOrder::kFlOrder, false>(state, gen);
}

// ============================================================================
// FastLanes + Frame-of-Reference Encode/Decode
// ============================================================================
//
// Round-trip is NOT flat: decoder produces output in transposed FL_ORDER
// (per 1024-block, output[t] == input[fromTransposed32(t)] + min). Chunked
// at 2048 values; per-chunk header stores [min, bit_width].

static void BM_FastLanesEncode(benchmark::State& state, Gen32 gen) {
  using ::arrow::util::fastlanes::FastLanesForCodec;
  // Round num_values up to a multiple of kChunkSize (2048) for this codec.
  int64_t num_values = state.range(0);
  num_values -= num_values % FastLanesForCodec::kChunkSize;
  if (num_values == 0) num_values = FastLanesForCodec::kChunkSize;

  auto values = gen(num_values);
  const int64_t uncompressed_size = num_values * sizeof(int32_t);

  std::vector<uint8_t> compressed(FastLanesForCodec::MaxEncodedSize(num_values));

  auto first = FastLanesForCodec::Encode(values.data(), num_values, compressed.data());
  ARROW_CHECK_OK(first.status());
  const int64_t comp_size = *first;

  for (auto _ : state) {
    auto sz = FastLanesForCodec::Encode(values.data(), num_values, compressed.data());
    benchmark::DoNotOptimize(sz);
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * uncompressed_size);
  state.SetItemsProcessed(state.iterations() * num_values);
  state.counters["compression_ratio"] =
      static_cast<double>(uncompressed_size) / static_cast<double>(comp_size);
}

static void BM_FastLanesDecode(benchmark::State& state, Gen32 gen) {
  using ::arrow::util::fastlanes::FastLanesForCodec;
  int64_t num_values = state.range(0);
  num_values -= num_values % FastLanesForCodec::kChunkSize;
  if (num_values == 0) num_values = FastLanesForCodec::kChunkSize;

  auto values = gen(num_values);
  const int64_t uncompressed_size = num_values * sizeof(int32_t);

  std::vector<uint8_t> compressed(FastLanesForCodec::MaxEncodedSize(num_values));
  auto comp = FastLanesForCodec::Encode(values.data(), num_values, compressed.data());
  ARROW_CHECK_OK(comp.status());
  const int64_t comp_size = *comp;

  std::vector<int32_t> decoded(num_values);
  for (auto _ : state) {
    auto status = FastLanesForCodec::Decode(decoded.data(), num_values,
                                             compressed.data(), comp_size);
    ARROW_CHECK_OK(status);
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * uncompressed_size);
  state.SetItemsProcessed(state.iterations() * num_values);
  state.counters["compression_ratio"] =
      static_cast<double>(uncompressed_size) / static_cast<double>(comp_size);
}

// Decode + FL_ORDER scatter: produces flat output in original input
// order (apples-to-apples vs PFOR/DeltaBitPack which also produce flat).
static void BM_FastLanesDecodeFlat(benchmark::State& state, Gen32 gen) {
  using ::arrow::util::fastlanes::FastLanesForCodec;
  int64_t num_values = state.range(0);
  num_values -= num_values % FastLanesForCodec::kChunkSize;
  if (num_values == 0) num_values = FastLanesForCodec::kChunkSize;

  auto values = gen(num_values);
  const int64_t uncompressed_size = num_values * sizeof(int32_t);

  std::vector<uint8_t> compressed(FastLanesForCodec::MaxEncodedSize(num_values));
  auto comp = FastLanesForCodec::Encode(values.data(), num_values, compressed.data());
  ARROW_CHECK_OK(comp.status());
  const int64_t comp_size = *comp;

  std::vector<int32_t> decoded(num_values);
  for (auto _ : state) {
    auto status = FastLanesForCodec::DecodeFlat(decoded.data(), num_values,
                                                 compressed.data(), comp_size);
    ARROW_CHECK_OK(status);
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * uncompressed_size);
  state.SetItemsProcessed(state.iterations() * num_values);
  state.counters["compression_ratio"] =
      static_cast<double>(uncompressed_size) / static_cast<double>(comp_size);
}

// ============================================================================
// Plain + ZSTD Encode/Decode
// ============================================================================

static void BM_PlainZstdEncode(benchmark::State& state, Gen32 gen) {
  const int64_t num_values = state.range(0);
  auto values = gen(num_values);
  const int64_t uncompressed_size = num_values * sizeof(int32_t);
  const uint8_t* raw = reinterpret_cast<const uint8_t*>(values.data());

  auto codec = *Codec::Create(Compression::ZSTD);
  int64_t max_comp = codec->MaxCompressedLen(uncompressed_size, raw);
  std::vector<uint8_t> compressed(max_comp);

  // Compute comp_size once for the counter
  int64_t comp_size =
      *codec->Compress(uncompressed_size, raw, max_comp, compressed.data());

  for (auto _ : state) {
    auto sz = *codec->Compress(uncompressed_size, raw, max_comp, compressed.data());
    benchmark::DoNotOptimize(sz);
  }

  state.SetBytesProcessed(state.iterations() * uncompressed_size);
  state.SetItemsProcessed(state.iterations() * num_values);
  state.counters["compression_ratio"] =
      static_cast<double>(uncompressed_size) / static_cast<double>(comp_size);
}

static void BM_PlainZstdDecode(benchmark::State& state, Gen32 gen) {
  const int64_t num_values = state.range(0);
  auto values = gen(num_values);
  const int64_t uncompressed_size = num_values * sizeof(int32_t);
  const uint8_t* raw = reinterpret_cast<const uint8_t*>(values.data());

  auto codec = *Codec::Create(Compression::ZSTD);
  int64_t max_comp = codec->MaxCompressedLen(uncompressed_size, raw);
  std::vector<uint8_t> compressed(max_comp);
  int64_t comp_size =
      *codec->Compress(uncompressed_size, raw, max_comp, compressed.data());

  std::vector<uint8_t> decompressed(uncompressed_size);
  for (auto _ : state) {
    auto result = codec->Decompress(comp_size, compressed.data(), uncompressed_size,
                                    decompressed.data());
    ARROW_CHECK_OK(result.status());
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * uncompressed_size);
  state.SetItemsProcessed(state.iterations() * num_values);
  state.counters["compression_ratio"] =
      static_cast<double>(uncompressed_size) / static_cast<double>(comp_size);
}

// ============================================================================
// Plain + LZ4 Encode/Decode
// ============================================================================

static void BM_PlainLz4Encode(benchmark::State& state, Gen32 gen) {
  const int64_t num_values = state.range(0);
  auto values = gen(num_values);
  const int64_t uncompressed_size = num_values * sizeof(int32_t);
  const uint8_t* raw = reinterpret_cast<const uint8_t*>(values.data());

  auto codec = *Codec::Create(Compression::LZ4_FRAME);
  int64_t max_comp = codec->MaxCompressedLen(uncompressed_size, raw);
  std::vector<uint8_t> compressed(max_comp);

  int64_t comp_size =
      *codec->Compress(uncompressed_size, raw, max_comp, compressed.data());

  for (auto _ : state) {
    auto sz = *codec->Compress(uncompressed_size, raw, max_comp, compressed.data());
    benchmark::DoNotOptimize(sz);
  }

  state.SetBytesProcessed(state.iterations() * uncompressed_size);
  state.SetItemsProcessed(state.iterations() * num_values);
  state.counters["compression_ratio"] =
      static_cast<double>(uncompressed_size) / static_cast<double>(comp_size);
}

static void BM_PlainLz4Decode(benchmark::State& state, Gen32 gen) {
  const int64_t num_values = state.range(0);
  auto values = gen(num_values);
  const int64_t uncompressed_size = num_values * sizeof(int32_t);
  const uint8_t* raw = reinterpret_cast<const uint8_t*>(values.data());

  auto codec = *Codec::Create(Compression::LZ4_FRAME);
  int64_t max_comp = codec->MaxCompressedLen(uncompressed_size, raw);
  std::vector<uint8_t> compressed(max_comp);
  int64_t comp_size =
      *codec->Compress(uncompressed_size, raw, max_comp, compressed.data());

  std::vector<uint8_t> decompressed(uncompressed_size);
  for (auto _ : state) {
    auto result = codec->Decompress(comp_size, compressed.data(), uncompressed_size,
                                    decompressed.data());
    ARROW_CHECK_OK(result.status());
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * uncompressed_size);
  state.SetItemsProcessed(state.iterations() * num_values);
  state.counters["compression_ratio"] =
      static_cast<double>(uncompressed_size) / static_cast<double>(comp_size);
}

// ============================================================================
// RleBitPackHybrid Encode/Decode
// ============================================================================

static void BM_RleBitPackEncode(benchmark::State& state, Gen32 gen) {
  const int64_t num_values = state.range(0);
  auto values = gen(num_values);
  const int64_t uncompressed_size = num_values * sizeof(int32_t);

  int32_t bit_width = ComputeBitWidth(values);
  int64_t max_buf =
      ::arrow::util::RleBitPackedEncoder::MaxBufferSize(bit_width, num_values) +
      ::arrow::util::RleBitPackedEncoder::MinBufferSize(bit_width);
  std::vector<uint8_t> buffer(max_buf);

  // Compute comp_size once for the counter
  int64_t comp_size;
  {
    ::arrow::util::RleBitPackedEncoder enc(buffer.data(),
                                           static_cast<int>(max_buf), bit_width);
    for (int64_t i = 0; i < num_values; ++i) {
      enc.Put(static_cast<uint64_t>(static_cast<uint32_t>(values[i])));
    }
    comp_size = enc.Flush();
  }

  for (auto _ : state) {
    ::arrow::util::RleBitPackedEncoder encoder(buffer.data(),
                                             static_cast<int>(max_buf), bit_width);
    for (int64_t i = 0; i < num_values; ++i) {
      encoder.Put(static_cast<uint64_t>(static_cast<uint32_t>(values[i])));
    }
    auto sz = encoder.Flush();
    benchmark::DoNotOptimize(sz);
  }

  state.SetBytesProcessed(state.iterations() * uncompressed_size);
  state.SetItemsProcessed(state.iterations() * num_values);
  state.counters["compression_ratio"] =
      static_cast<double>(uncompressed_size) / static_cast<double>(comp_size);
}

static void BM_RleBitPackDecode(benchmark::State& state, Gen32 gen) {
  const int64_t num_values = state.range(0);
  auto values = gen(num_values);
  const int64_t uncompressed_size = num_values * sizeof(int32_t);

  int32_t bit_width = ComputeBitWidth(values);
  int64_t max_buf =
      ::arrow::util::RleBitPackedEncoder::MaxBufferSize(bit_width, num_values) +
      ::arrow::util::RleBitPackedEncoder::MinBufferSize(bit_width);
  std::vector<uint8_t> buffer(max_buf);

  ::arrow::util::RleBitPackedEncoder encoder(buffer.data(),
                                           static_cast<int>(max_buf), bit_width);
  for (int64_t i = 0; i < num_values; ++i) {
    encoder.Put(static_cast<uint64_t>(static_cast<uint32_t>(values[i])));
  }
  int comp_size = encoder.Flush();

  std::vector<int32_t> decoded(num_values);
  for (auto _ : state) {
    ::arrow::util::RleBitPackedParser parser(buffer.data(), comp_size, bit_width);
    int64_t out_idx = 0;
    struct Handler {
      int32_t* output;
      int64_t* idx;
      int64_t max_values;
      int32_t bw;
      ::arrow::util::RleBitPackedParser::ControlFlow OnRleRun(
          ::arrow::util::RleRun run) {
        ::arrow::util::RleRunDecoder<int32_t> dec(run, bw);
        auto want = static_cast<int32_t>(
            std::min(static_cast<int64_t>(run.values_count()), max_values - *idx));
        auto count = dec.GetBatch(output + *idx, want, bw);
        *idx += count;
        return *idx >= max_values
                   ? ::arrow::util::RleBitPackedParser::ControlFlow::Break
                   : ::arrow::util::RleBitPackedParser::ControlFlow::Continue;
      }
      ::arrow::util::RleBitPackedParser::ControlFlow OnBitPackedRun(
          ::arrow::util::BitPackedRun run) {
        ::arrow::util::BitPackedRunDecoder<int32_t> dec(run, bw);
        auto want = static_cast<int32_t>(
            std::min(static_cast<int64_t>(dec.remaining()), max_values - *idx));
        auto count = dec.GetBatch(output + *idx, want, bw);
        *idx += count;
        return *idx >= max_values
                   ? ::arrow::util::RleBitPackedParser::ControlFlow::Break
                   : ::arrow::util::RleBitPackedParser::ControlFlow::Continue;
      }
    };
    Handler handler{decoded.data(), &out_idx, num_values, bit_width};
    parser.Parse(handler);
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * uncompressed_size);
  state.SetItemsProcessed(state.iterations() * num_values);
  state.counters["compression_ratio"] =
      static_cast<double>(uncompressed_size) / static_cast<double>(comp_size);
}

// ============================================================================
// ByteStreamSplit + Codec (ZSTD or LZ4)
// ============================================================================

static void BM_BssCodecEncode(benchmark::State& state, Gen32 gen,
                              Compression::type codec_type) {
  const int64_t num_values = state.range(0);
  auto values = gen(num_values);
  const int64_t uncompressed_size = num_values * sizeof(int32_t);

  auto descr = MakeInt32Descriptor();
  auto encoder = MakeTypedEncoder<Int32Type>(Encoding::BYTE_STREAM_SPLIT,
                                            /*use_dictionary=*/false, descr.get());
  auto codec = *Codec::Create(codec_type);

  encoder->Put(values.data(), static_cast<int>(num_values));
  auto encoded_buf = encoder->FlushValues();
  int64_t encoded_size = encoded_buf->size();

  int64_t max_comp = codec->MaxCompressedLen(encoded_size, encoded_buf->data());
  std::vector<uint8_t> compressed(max_comp);

  // Compute comp_size once for the counter
  int64_t comp_size =
      *codec->Compress(encoded_size, encoded_buf->data(), max_comp, compressed.data());

  for (auto _ : state) {
    encoder->Put(values.data(), static_cast<int>(num_values));
    auto buf = encoder->FlushValues();
    auto sz =
        *codec->Compress(buf->size(), buf->data(), max_comp, compressed.data());
    benchmark::DoNotOptimize(sz);
  }

  state.SetBytesProcessed(state.iterations() * uncompressed_size);
  state.SetItemsProcessed(state.iterations() * num_values);
  state.counters["compression_ratio"] =
      static_cast<double>(uncompressed_size) / static_cast<double>(comp_size);
}

static void BM_BssCodecDecode(benchmark::State& state, Gen32 gen,
                              Compression::type codec_type) {
  const int64_t num_values = state.range(0);
  auto values = gen(num_values);
  const int64_t uncompressed_size = num_values * sizeof(int32_t);

  auto descr = MakeInt32Descriptor();
  auto encoder = MakeTypedEncoder<Int32Type>(Encoding::BYTE_STREAM_SPLIT,
                                            /*use_dictionary=*/false, descr.get());
  auto codec = *Codec::Create(codec_type);

  encoder->Put(values.data(), static_cast<int>(num_values));
  auto encoded_buf = encoder->FlushValues();
  int64_t encoded_size = encoded_buf->size();

  int64_t max_comp = codec->MaxCompressedLen(encoded_size, encoded_buf->data());
  std::vector<uint8_t> compressed(max_comp);
  int64_t comp_size =
      *codec->Compress(encoded_size, encoded_buf->data(), max_comp, compressed.data());

  std::vector<uint8_t> decompressed(encoded_size);
  std::vector<int32_t> decoded(num_values);
  auto decoder = MakeTypedDecoder<Int32Type>(Encoding::BYTE_STREAM_SPLIT, descr.get());

  for (auto _ : state) {
    auto result = codec->Decompress(comp_size, compressed.data(), encoded_size,
                                    decompressed.data());
    ARROW_CHECK_OK(result.status());
    decoder->SetData(static_cast<int>(num_values), decompressed.data(),
                     static_cast<int>(encoded_size));
    decoder->Decode(decoded.data(), static_cast<int>(num_values));
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * uncompressed_size);
  state.SetItemsProcessed(state.iterations() * num_values);
  state.counters["compression_ratio"] =
      static_cast<double>(uncompressed_size) / static_cast<double>(comp_size);
}

// Wrappers for BSS+ZSTD
static void BM_BssZstdEncode(benchmark::State& state, Gen32 gen) {
  BM_BssCodecEncode(state, gen, Compression::ZSTD);
}
static void BM_BssZstdDecode(benchmark::State& state, Gen32 gen) {
  BM_BssCodecDecode(state, gen, Compression::ZSTD);
}

// Wrappers for BSS+LZ4
static void BM_BssLz4Encode(benchmark::State& state, Gen32 gen) {
  BM_BssCodecEncode(state, gen, Compression::LZ4_FRAME);
}
static void BM_BssLz4Decode(benchmark::State& state, Gen32 gen) {
  BM_BssCodecDecode(state, gen, Compression::LZ4_FRAME);
}

// ============================================================================
// Layout against layout at equal implementation effort
//
// The codec rows in §3 compare Arrow's sequential unpack, which is hand-written
// per-ISA intrinsics reached through an out-of-line runtime dispatch, against a
// portable width-templated header template. Two variables move together, so
// neither number isolates the layout. This pins one of them:
//
//   sequential   ScalarUnpackerForWidth<uint32_t, W>, the generated portable
//                scalar kernel, width a template parameter, inlinable.
//   interleaved  UnpackBlock<W>, likewise portable, templated and inlinable.
//
// Same source language, same dispatch structure, same output buffer, same bits
// per value. What differs is where a value sits, which is the part a file
// format would have to specify.
// ============================================================================

template <uint32_t kWidth>
void BM_KernelSequentialScalar(benchmark::State& state) {
  using Unpacker = ::arrow::internal::ScalarUnpackerForWidth<uint32_t, kWidth>;
  constexpr int kPerCall = Unpacker::kValuesUnpacked;
  constexpr int kBytesPerCall = Unpacker::kBytesRead;

  int64_t num_values = state.range(0);
  num_values -= num_values % kPerCall;
  const int64_t ncalls = num_values / kPerCall;

  std::mt19937 rng(909);
  std::vector<uint8_t> packed(ncalls * kBytesPerCall + 64, 0);
  for (auto& b : packed) b = static_cast<uint8_t>(rng());
  std::vector<uint32_t> out(num_values, 0);

  for (auto _ : state) {
    const uint8_t* in = packed.data();
    uint32_t* dst = out.data();
    for (int64_t c = 0; c < ncalls; ++c) {
      in = Unpacker::unpack(in, dst);
      dst += kPerCall;
    }
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * num_values *
                          static_cast<int64_t>(sizeof(uint32_t)));
  state.counters["bit_width"] = kWidth;
}

template <uint32_t kWidth>
void BM_KernelInterleavedScalar(benchmark::State& state) {
  constexpr int64_t kPerCall = ::arrow::util::fastlanes::kBlockSize;
  constexpr int64_t kWordsPerCall = kWidth * ::arrow::util::fastlanes::kLanes;

  int64_t num_values = state.range(0);
  num_values -= num_values % kPerCall;
  const int64_t ncalls = num_values / kPerCall;

  std::mt19937 rng(909);
  std::vector<uint32_t> packed(ncalls * kWordsPerCall, 0);
  for (auto& w : packed) w = rng();
  std::vector<uint32_t> out(num_values, 0);

  for (auto _ : state) {
    for (int64_t c = 0; c < ncalls; ++c) {
      ::arrow::util::fastlanes::UnpackBlock<kWidth>(packed.data() + c * kWordsPerCall,
                                                   out.data() + c * kPerCall);
    }
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * num_values *
                          static_cast<int64_t>(sizeof(uint32_t)));
  state.counters["bit_width"] = kWidth;
}

// A third leg, for reading alongside the two above rather than against one of
// them. The pair above pins language, dispatch structure and portability so the
// only remaining variable is where a value sits. This one deliberately unpins
// them, because it is what the codec rows actually execute: Arrow's shipping
// sequential unpacker, runtime bit width, reached through the out-of-line
// dispatch, one call per 1024-value vector exactly as PforCompression calls it.
//
// Reading all three separates two questions the codec rows answer together. The
// gap between this leg and the interleaved one is what a reader that already has
// a vectorized unpacker would gain from the layout. The gap between the two
// portable legs is what a reader that has no such unpacker would gain. Those are
// different audiences and a format decision serves both, so neither number alone
// is the answer.
template <uint32_t kWidth>
void BM_KernelSequentialArrowSimd(benchmark::State& state) {
  constexpr int64_t kPerCall = ::arrow::util::fastlanes::kBlockSize;
  // 1024 * kWidth bits is always a whole number of bytes, and a multiple of 128,
  // so every per-call offset lands on a vector boundary as it does in the codec.
  constexpr int64_t kBytesPerCall = kPerCall * kWidth / 8;

  int64_t num_values = state.range(0);
  num_values -= num_values % kPerCall;
  const int64_t ncalls = num_values / kPerCall;

  std::mt19937 rng(909);
  std::vector<uint8_t> packed(ncalls * kBytesPerCall + 64, 0);
  for (auto& b : packed) b = static_cast<uint8_t>(rng());
  std::vector<uint32_t> out(num_values, 0);

  for (auto _ : state) {
    for (int64_t c = 0; c < ncalls; ++c) {
      ::arrow::internal::unpack(packed.data() + c * kBytesPerCall,
                                out.data() + c * kPerCall,
                                ::arrow::internal::UnpackOptions{
                                    static_cast<int>(kPerCall),
                                    static_cast<int>(kWidth)});
    }
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * num_values *
                          static_cast<int64_t>(sizeof(uint32_t)));
  state.counters["bit_width"] = kWidth;
}

// ============================================================================
// Benchmark Registration
// ============================================================================

static void CustomArgs(benchmark::internal::Benchmark* b) { b->Arg(102400); }

// Macro to register all algorithms for a given dataset
#define REGISTER_DATASET(Name, GenFunc)                                          \
  BENCHMARK_CAPTURE(BM_PforEncode, Name, &GenFunc)->Apply(CustomArgs);          \
  BENCHMARK_CAPTURE(BM_PforDecode, Name, &GenFunc)->Apply(CustomArgs);          \
  BENCHMARK_CAPTURE(BM_PforFastLanesEncode, Name, &GenFunc)->Apply(CustomArgs); \
  BENCHMARK_CAPTURE(BM_PforFastLanesDecode, Name, &GenFunc)->Apply(CustomArgs); \
  BENCHMARK_CAPTURE(BM_PforFastLanesDecodeTransposed, Name, &GenFunc)->Apply(CustomArgs); \
  BENCHMARK_CAPTURE(BM_PforFastLanesOrderedEncode, Name, &GenFunc)->Apply(CustomArgs); \
  BENCHMARK_CAPTURE(BM_PforFastLanesOrderedDecode, Name, &GenFunc)->Apply(CustomArgs); \
  BENCHMARK_CAPTURE(BM_DeltaBitPackEncode, Name, &GenFunc)->Apply(CustomArgs);  \
  BENCHMARK_CAPTURE(BM_DeltaBitPackDecode, Name, &GenFunc)->Apply(CustomArgs);  \
  BENCHMARK_CAPTURE(BM_LaneDeltaEncode, Name, &GenFunc)->Apply(CustomArgs);     \
  BENCHMARK_CAPTURE(BM_LaneDeltaDecode, Name, &GenFunc)->Apply(CustomArgs);     \
  BENCHMARK_CAPTURE(BM_LaneDeltaFlOrderEncode, Name, &GenFunc)->Apply(CustomArgs); \
  BENCHMARK_CAPTURE(BM_LaneDeltaFlOrderDecode, Name, &GenFunc)->Apply(CustomArgs); \
  BENCHMARK_CAPTURE(BM_LaneDeltaFlOrderDecodeTransposed, Name, &GenFunc)->Apply(CustomArgs); \
  BENCHMARK_CAPTURE(BM_LaneDeltaWidthStudy, Name, &GenFunc)->Apply(CustomArgs);  \
  BENCHMARK_CAPTURE(BM_FastLanesEncode, Name, &GenFunc)->Apply(CustomArgs);     \
  BENCHMARK_CAPTURE(BM_FastLanesDecode, Name, &GenFunc)->Apply(CustomArgs);     \
  BENCHMARK_CAPTURE(BM_FastLanesDecodeFlat, Name, &GenFunc)->Apply(CustomArgs); \
  BENCHMARK_CAPTURE(BM_PlainZstdEncode, Name, &GenFunc)->Apply(CustomArgs);     \
  BENCHMARK_CAPTURE(BM_PlainZstdDecode, Name, &GenFunc)->Apply(CustomArgs);     \
  BENCHMARK_CAPTURE(BM_PlainLz4Encode, Name, &GenFunc)->Apply(CustomArgs);      \
  BENCHMARK_CAPTURE(BM_PlainLz4Decode, Name, &GenFunc)->Apply(CustomArgs);      \
  BENCHMARK_CAPTURE(BM_RleBitPackEncode, Name, &GenFunc)->Apply(CustomArgs);    \
  BENCHMARK_CAPTURE(BM_RleBitPackDecode, Name, &GenFunc)->Apply(CustomArgs);    \
  BENCHMARK_CAPTURE(BM_BssZstdEncode, Name, &GenFunc)->Apply(CustomArgs);       \
  BENCHMARK_CAPTURE(BM_BssZstdDecode, Name, &GenFunc)->Apply(CustomArgs);       \
  BENCHMARK_CAPTURE(BM_BssLz4Encode, Name, &GenFunc)->Apply(CustomArgs);        \
  BENCHMARK_CAPTURE(BM_BssLz4Decode, Name, &GenFunc)->Apply(CustomArgs);

// ClickBench datasets
REGISTER_DATASET(ClientIP, GenClientIP)
REGISTER_DATASET(UrlRegionID, GenUrlRegionID)
REGISTER_DATASET(CounterID, GenCounterID)
REGISTER_DATASET(EventDate, GenEventDate)
REGISTER_DATASET(EventTime, GenEventTime)
REGISTER_DATASET(GoodEvent, GenGoodEvent)
REGISTER_DATASET(HID, GenHID)
REGISTER_DATASET(HitColor, GenHitColor)
REGISTER_DATASET(IPNetworkID, GenIPNetworkID)
REGISTER_DATASET(JavaEnable, GenJavaEnable)
REGISTER_DATASET(OS, GenOS)
REGISTER_DATASET(Resolution, GenResolution)
REGISTER_DATASET(TrafficSourceID, GenTrafficSourceID)
REGISTER_DATASET(UserAgent, GenUserAgent)

// TPC-DS datasets
REGISTER_DATASET(TpcdsSoldDateSk, GenTpcdsSoldDateSk)
REGISTER_DATASET(TpcdsStoreSk, GenTpcdsStoreSk)
REGISTER_DATASET(TpcdsItemSk, GenTpcdsItemSk)
REGISTER_DATASET(TpcdsQuantity, GenTpcdsQuantity)
REGISTER_DATASET(TpcdsCustomerSk, GenTpcdsCustomerSk)
REGISTER_DATASET(TpcdsExtSalesPrice, GenTpcdsExtSalesPrice)
REGISTER_DATASET(TpcdsNetProfit, GenTpcdsNetProfit)
REGISTER_DATASET(TpcdsDYear, GenTpcdsDYear)
// TPC-H datasets
REGISTER_DATASET(TpchLQuantity, GenTpchLQuantity)
REGISTER_DATASET(TpchLExtendedPrice, GenTpchLExtendedPrice)
REGISTER_DATASET(TpchLDiscount, GenTpchLDiscount)
REGISTER_DATASET(TpchLShipDate, GenTpchLShipDate)
// NYC taxi datasets
// Sorted / near-sorted: the shapes delta encoding exists for
REGISTER_DATASET(SortedUnixTime, GenSortedUnixTime)
REGISTER_DATASET(SortedKeyDups, GenSortedKeyDups)
REGISTER_DATASET(MonotoneRowId, GenMonotoneRowId)
// Registered only here: the hand-built stream needs a constant stride.
BENCHMARK_CAPTURE(BM_DeltaBitPackBigBlockDecode, MonotoneRowId, &GenMonotoneRowId)
    ->Apply(CustomArgs);
REGISTER_DATASET(NearSortedUnixTime, GenNearSortedUnixTime)

REGISTER_DATASET(TaxiPickupUnixTime, GenTaxiPickupUnixTime)
REGISTER_DATASET(TaxiTripDistanceX100, GenTaxiTripDistanceX100)
REGISTER_DATASET(TaxiFareCents, GenTaxiFareCents)


// The layout-vs-layout kernel comparison: every width, both sides portable
// scalar. 102 400 values so the output is the same 400 KiB working set the
// codec benchmarks use. Widths 1-31; at 32 there is nothing to unpack.
#define REGISTER_KERNEL_WIDTH(W)                                            \
  BENCHMARK_TEMPLATE(BM_KernelSequentialScalar, W)->Arg(102400);            \
  BENCHMARK_TEMPLATE(BM_KernelSequentialArrowSimd, W)->Arg(102400);         \
  BENCHMARK_TEMPLATE(BM_KernelInterleavedScalar, W)->Arg(102400);

REGISTER_KERNEL_WIDTH(1)
REGISTER_KERNEL_WIDTH(2)
REGISTER_KERNEL_WIDTH(3)
REGISTER_KERNEL_WIDTH(4)
REGISTER_KERNEL_WIDTH(5)
REGISTER_KERNEL_WIDTH(6)
REGISTER_KERNEL_WIDTH(7)
REGISTER_KERNEL_WIDTH(8)
REGISTER_KERNEL_WIDTH(9)
REGISTER_KERNEL_WIDTH(10)
REGISTER_KERNEL_WIDTH(11)
REGISTER_KERNEL_WIDTH(12)
REGISTER_KERNEL_WIDTH(13)
REGISTER_KERNEL_WIDTH(14)
REGISTER_KERNEL_WIDTH(15)
REGISTER_KERNEL_WIDTH(16)
REGISTER_KERNEL_WIDTH(17)
REGISTER_KERNEL_WIDTH(18)
REGISTER_KERNEL_WIDTH(19)
REGISTER_KERNEL_WIDTH(20)
REGISTER_KERNEL_WIDTH(21)
REGISTER_KERNEL_WIDTH(22)
REGISTER_KERNEL_WIDTH(23)
REGISTER_KERNEL_WIDTH(24)
REGISTER_KERNEL_WIDTH(25)
REGISTER_KERNEL_WIDTH(26)
REGISTER_KERNEL_WIDTH(27)
REGISTER_KERNEL_WIDTH(28)
REGISTER_KERNEL_WIDTH(29)
REGISTER_KERNEL_WIDTH(30)
REGISTER_KERNEL_WIDTH(31)
// Width 32 has no sequential specialization: a full-width value needs no
// unpacking, so both layouts degenerate to a copy.

}  // namespace
}  // namespace parquet
