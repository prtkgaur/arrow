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

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "arrow/testing/gtest_util.h"
#include "arrow/util/logging.h"
#include "arrow/util/pfor/pfor_internal.h"
#include "arrow/util/pfor/pfor_wrapper_internal.h"

namespace arrow::util::pfor {

// ======================================================================
// Test fixture
//
// PFOR is instantiated for int32_t and int64_t, and the two differ in the
// widths of the frame of reference, the packed deltas and the exception values,
// so nearly every test here belongs to both. The helpers hide the width
// arithmetic; tests that poke at particular byte offsets derive them from
// sizeof(T) and kStoredSize rather than writing them out.

template <typename T>
class PforTest : public ::testing::Test {
 protected:
  using UnsignedT = typename PforTypeTraits<T>::UnsignedType;

  /// Encode, serialize and decode a single vector, checking that it round-trips.
  ///
  /// \param[out] encoded_out the encoded form, for tests that assert on its
  ///             metadata; may be null
  void RoundTripVector(const std::vector<T>& values,
                       PforEncodedVector<T>* encoded_out = nullptr) {
    const auto num_elements = static_cast<int32_t>(values.size());
    auto encoded = PforCompression<T>::EncodeVector(values.data(), num_elements);

    const int64_t needed =
        PforCompression<T>::SerializedVectorSize(encoded, num_elements);
    std::vector<uint8_t> buffer(needed);
    ASSERT_OK_AND_ASSIGN(const int64_t written, PforCompression<T>::SerializeVector(
                                                    encoded, num_elements, buffer));
    ASSERT_EQ(needed, written);

    std::vector<T> decoded(values.size());
    ASSERT_OK(PforCompression<T>::DecodeVector(buffer, num_elements, decoded.data()));
    ASSERT_EQ(values, decoded);

    if (encoded_out != nullptr) {
      *encoded_out = std::move(encoded);
    }
  }

  /// Encode and decode a whole page, checking that it round-trips and stays
  /// inside the bound the encoder allocates from.
  ///
  /// \param[out] comp_size_out the compressed size, for tests that assert on
  ///             the compression ratio; may be null
  void RoundTripPage(const std::vector<T>& values, int64_t* comp_size_out = nullptr) {
    const auto num_values = static_cast<int32_t>(values.size());
    ASSERT_OK_AND_ASSIGN(const int64_t max_size,
                         PforWrapper<T>::GetMaxCompressedSize(num_values));
    std::vector<uint8_t> compressed(max_size);
    int64_t comp_size = max_size;
    ASSERT_OK(
        PforWrapper<T>::Encode(values.data(), num_values, compressed.data(), &comp_size));
    ASSERT_GT(comp_size, 0);
    ASSERT_LE(comp_size, max_size);

    std::vector<T> decoded(values.size());
    ASSERT_OK(
        PforWrapper<T>::Decode(compressed.data(), comp_size, num_values, decoded.data()));
    ASSERT_EQ(values, decoded);

    if (comp_size_out != nullptr) {
      *comp_size_out = comp_size;
    }
  }

  /// Encode `values` and return the compressed bytes, sized exactly.
  static std::vector<uint8_t> EncodePage(const std::vector<T>& values) {
    const auto num_values = static_cast<int32_t>(values.size());
    int64_t comp_size = PforWrapper<T>::GetMaxCompressedSize(num_values).ValueOrDie();
    std::vector<uint8_t> compressed(comp_size);
    ARROW_CHECK_OK(
        PforWrapper<T>::Encode(values.data(), num_values, compressed.data(), &comp_size));
    compressed.resize(comp_size);
    return compressed;
  }

  /// Offset of the first vector in a single-vector page: header, then the
  /// one-entry offset array.
  static constexpr int64_t kFirstVectorOffset =
      PforConstants::kHeaderSize +
      static_cast<int64_t>(sizeof(PforConstants::OffsetType));

  /// Offsets of the PforVectorInfo fields, relative to the start of a vector.
  static constexpr int64_t kBitWidthOffset = static_cast<int64_t>(sizeof(T));
  static constexpr int64_t kNumExceptionsOffset = kBitWidthOffset + 1;

  static std::vector<T> RandomValues(int32_t num_values, T min, T max, uint32_t seed) {
    std::vector<T> values(num_values);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<T> dist(min, max);
    for (auto& v : values) v = dist(rng);
    return values;
  }
};

using PforTypes = ::testing::Types<int32_t, int64_t>;
TYPED_TEST_SUITE(PforTest, PforTypes);

namespace {

void StoreLE32(uint8_t* dest, uint32_t value) {
  for (int i = 0; i < 4; ++i) {
    dest[i] = static_cast<uint8_t>(value >> (8 * i));
  }
}

}  // namespace

// ======================================================================
// Constants Tests

TEST(PforConstantsTest, VectorSizeIsPowerOfTwo) {
  EXPECT_EQ(PforConstants::kPforVectorSize, 1024);
  EXPECT_EQ(1 << PforConstants::kDefaultLogVectorSize, PforConstants::kPforVectorSize);
}

TEST(PforConstantsTest, VectorInfoSizes) {
  EXPECT_EQ(PforTypeTraits<int32_t>::kVectorInfoSize, 7);
  EXPECT_EQ(PforTypeTraits<int64_t>::kVectorInfoSize, 11);
}

// ======================================================================
// BitsRequired Tests

TYPED_TEST(PforTest, BitsRequired) {
  using T = TypeParam;
  using UnsignedT = typename PforTypeTraits<T>::UnsignedType;

  EXPECT_EQ(PforTypeTraits<T>::BitsRequired(0), 0);
  EXPECT_EQ(PforTypeTraits<T>::BitsRequired(1), 1);
  EXPECT_EQ(PforTypeTraits<T>::BitsRequired(2), 2);
  EXPECT_EQ(PforTypeTraits<T>::BitsRequired(3), 2);
  EXPECT_EQ(PforTypeTraits<T>::BitsRequired(255), 8);
  EXPECT_EQ(PforTypeTraits<T>::BitsRequired(256), 9);
  EXPECT_EQ(PforTypeTraits<T>::BitsRequired(std::numeric_limits<UnsignedT>::max()),
            PforTypeTraits<T>::kMaxBitWidth);
}

// ======================================================================
// VectorInfo Serialization Tests

TYPED_TEST(PforTest, VectorInfoRoundTrip) {
  using T = TypeParam;
  PforVectorInfo<T> info;
  info.set_frame_of_reference(-42);
  info.set_bit_width(17);
  info.set_num_exceptions(300);

  std::vector<uint8_t> buf(PforVectorInfo<T>::kStoredSize);
  info.Store(buf);
  ASSERT_OK_AND_ASSIGN(auto loaded, PforVectorInfo<T>::Load(buf));

  EXPECT_EQ(loaded.frame_of_reference(), -42);
  EXPECT_EQ(loaded.bit_width(), 17);
  EXPECT_EQ(loaded.num_exceptions(), 300);
}

// The extremes of every field at once: the widest frame of reference the type
// can hold and the full bit width. A six-bit width field could not store 64,
// and reported such a vector as constant.
TYPED_TEST(PforTest, VectorInfoRoundTripAtFieldExtremes) {
  using T = TypeParam;
  constexpr T kFrameOfReference = std::numeric_limits<T>::min();
  PforVectorInfo<T> info(kFrameOfReference, PforTypeTraits<T>::kMaxBitWidth, 30000);

  std::vector<uint8_t> buf(PforVectorInfo<T>::kStoredSize);
  info.Store(buf);
  ASSERT_OK_AND_ASSIGN(auto loaded, PforVectorInfo<T>::Load(buf));

  EXPECT_EQ(loaded.frame_of_reference(), kFrameOfReference);
  EXPECT_EQ(loaded.bit_width(), PforTypeTraits<T>::kMaxBitWidth);
  EXPECT_EQ(loaded.num_exceptions(), 30000);
}

// Every element of a maximum-size vector can be an exception, so the count has
// to reach kMaxVectorSize. A signed 16-bit field stopped one short of that.
TYPED_TEST(PforTest, VectorInfoNumExceptionsAtMaxVectorSize) {
  using T = TypeParam;
  PforVectorInfo<T> info;
  info.set_bit_width(3);
  info.set_num_exceptions(PforConstants::kMaxVectorSize);

  std::vector<uint8_t> buf(PforVectorInfo<T>::kStoredSize);
  info.Store(buf);
  ASSERT_OK_AND_ASSIGN(auto loaded, PforVectorInfo<T>::Load(buf));

  EXPECT_EQ(loaded.num_exceptions(), PforConstants::kMaxVectorSize);
}

TYPED_TEST(PforTest, VectorInfoRejectsNumExceptionsAboveMaxVectorSize) {
  using T = TypeParam;
  // [FOR] [bit_width 1B] [num_exceptions 2B], with a count no vector can have.
  std::vector<uint8_t> buf(PforVectorInfo<T>::kStoredSize, 0);
  buf[TestFixture::kBitWidthOffset] = 3;
  buf[TestFixture::kNumExceptionsOffset] = 0xFF;
  buf[TestFixture::kNumExceptionsOffset + 1] = 0xFF;
  ASSERT_RAISES(Invalid, PforVectorInfo<T>::Load(buf));
}

TYPED_TEST(PforTest, VectorInfoRejectsUndersizedBuffer) {
  using T = TypeParam;
  std::vector<uint8_t> buf(PforVectorInfo<T>::kStoredSize - 1, 0);
  ASSERT_RAISES(Invalid, PforVectorInfo<T>::Load(buf));
}

// ======================================================================
// Cost Model Tests

TYPED_TEST(PforTest, CostModelAllIdentical) {
  using T = TypeParam;
  using UnsignedT = typename PforTypeTraits<T>::UnsignedType;
  // All deltas are 0 => bit_width should be 0, no exceptions
  std::vector<UnsignedT> deltas(100, 0);
  auto result = PforCompression<T>::FindOptimalBitWidth(deltas.data(), 100);
  EXPECT_EQ(result.bit_width, 0);
  EXPECT_EQ(result.num_exceptions, 0);
}

TYPED_TEST(PforTest, CostModelSingleOutlier) {
  using T = TypeParam;
  using UnsignedT = typename PforTypeTraits<T>::UnsignedType;
  // 99 values fit in 3 bits, 1 outlier needs 16 bits
  std::vector<UnsignedT> deltas(100, 5);  // all fit in 3 bits
  deltas[50] = 50000;                     // outlier: 16 bits
  auto result = PforCompression<T>::FindOptimalBitWidth(deltas.data(), 100);
  // An exception costs a 16-bit position plus a full-width value, so at
  // bit_width=3 the vector costs 100*3 + 1*(16 + 8*sizeof(T)) bits, against
  // 100*16 at bit_width=16. Patching the one outlier wins for both widths of T.
  EXPECT_EQ(result.bit_width, 3);
  EXPECT_EQ(result.num_exceptions, 1);
}

TYPED_TEST(PforTest, CostModelNoOutliers) {
  using T = TypeParam;
  using UnsignedT = typename PforTypeTraits<T>::UnsignedType;
  // All values fit in 8 bits
  std::vector<UnsignedT> deltas(100);
  for (int32_t i = 0; i < 100; ++i) deltas[i] = i * 2;
  auto result = PforCompression<T>::FindOptimalBitWidth(deltas.data(), 100);
  EXPECT_EQ(result.num_exceptions, 0);
  EXPECT_LE(result.bit_width, 8);
}

// ======================================================================
// Vector Encode/Decode Round-Trip Tests

TYPED_TEST(PforTest, VectorSimpleSequence) {
  using T = TypeParam;
  std::vector<T> values(64);
  std::iota(values.begin(), values.end(), 100);

  PforEncodedVector<T> encoded;
  this->RoundTripVector(values, &encoded);
  EXPECT_EQ(encoded.info().frame_of_reference(), 100);
  EXPECT_EQ(encoded.info().num_exceptions(), 0);
}

TYPED_TEST(PforTest, VectorWithOutlier) {
  using T = TypeParam;
  std::vector<T> values = {100, 102, 101, 103, 100, 99, 50000, 104};

  PforEncodedVector<T> encoded;
  this->RoundTripVector(values, &encoded);
  EXPECT_EQ(encoded.info().frame_of_reference(), 99);
  EXPECT_GT(encoded.info().num_exceptions(), 0);
}

TYPED_TEST(PforTest, VectorAllIdentical) {
  using T = TypeParam;
  std::vector<T> values(100, 42);

  PforEncodedVector<T> encoded;
  this->RoundTripVector(values, &encoded);
  EXPECT_EQ(encoded.info().bit_width(), 0);
  EXPECT_EQ(encoded.info().num_exceptions(), 0);
}

TYPED_TEST(PforTest, VectorNegativeValues) {
  using T = TypeParam;
  std::vector<T> values = {-100, -50, -200, -1, -150};

  PforEncodedVector<T> encoded;
  this->RoundTripVector(values, &encoded);
  EXPECT_EQ(encoded.info().frame_of_reference(), -200);
}

// The frame of reference is subtracted in the unsigned domain, so a vector
// spanning the whole range of T has a delta that only wraps to the right answer.
TYPED_TEST(PforTest, VectorMinMaxEdge) {
  using T = TypeParam;
  std::vector<T> values = {std::numeric_limits<T>::min(), std::numeric_limits<T>::max(),
                           0, -1, 1};
  this->RoundTripVector(values);
}

TYPED_TEST(PforTest, VectorSingleElement) {
  using T = TypeParam;
  this->RoundTripVector(std::vector<T>{42});
}

// ======================================================================
// Page-Level Wrapper Tests

TYPED_TEST(PforTest, PageSmall) { this->RoundTripPage({10, 20, 30, 40, 50}); }

TYPED_TEST(PforTest, PageExactOneVector) {
  using T = TypeParam;
  std::vector<T> values(PforConstants::kPforVectorSize);
  std::iota(values.begin(), values.end(), 0);
  this->RoundTripPage(values);
}

TYPED_TEST(PforTest, PageMultipleVectors) {
  // 2.5 vectors worth of data
  this->RoundTripPage(TestFixture::RandomValues(2560, 0, 1000, /*seed=*/42));
}

TYPED_TEST(PforTest, PageWithOutliers) {
  using T = TypeParam;
  std::vector<T> values(1024, 100);
  values[0] = -999999;
  values[100] = 888888;
  values[500] = 777777;
  values[1023] = -123456;
  this->RoundTripPage(values);
}

TYPED_TEST(PforTest, PagePartialTrailingVector) {
  using T = TypeParam;
  auto values = TestFixture::RandomValues(3000, 0, 100000, /*seed=*/123);
  values[0] = std::numeric_limits<T>::max() / 2;
  values[1500] = std::numeric_limits<T>::min() / 2;
  this->RoundTripPage(values);
}

TYPED_TEST(PforTest, PageSingleElement) {
  using T = TypeParam;
  this->RoundTripPage(std::vector<T>{42});
}

TYPED_TEST(PforTest, PageAllZeros) {
  using T = TypeParam;
  std::vector<T> values(1024, 0);
  int64_t comp_size = 0;
  this->RoundTripPage(values, &comp_size);
  // bit_width is 0, so the page is header, offset array and vector info only.
  EXPECT_LT(comp_size, 100);
}

TYPED_TEST(PforTest, PageLargeRandom) {
  using T = TypeParam;
  this->RoundTripPage(TestFixture::RandomValues(10000, std::numeric_limits<T>::min(),
                                                std::numeric_limits<T>::max(),
                                                /*seed=*/99));
}

// Covers the FOR==0 fast path (bit_width > 0 with frame_of_reference == 0):
// decode unpacks straight into the output, skipping the scratch buffer and the
// add-FOR pass. Forces min == 0, a range that needs bit_width > 0, outliers to
// exercise the exception patch on that path, and multiple vectors.
TYPED_TEST(PforTest, PageZeroMinWithExceptions) {
  auto values = TestFixture::RandomValues(3000, 0, 500, /*seed=*/2024);
  values[0] = 0;            // force min == 0 -> FOR == 0
  values[100] = 1'000'000;  // outliers -> exceptions on the FOR==0 path
  values[1500] = 999'999;
  values[2999] = 500'000;
  this->RoundTripPage(values);
}

// ======================================================================
// Encode Argument Validation

TYPED_TEST(PforTest, EncodeZeroValuesWritesABareHeader) {
  using T = TypeParam;
  // An all-null page holds no values, and the reader still has to be able to load
  // a header from it, so zero values encodes to the header and nothing else. The
  // pointer is null because the buffer the values would come from is empty.
  std::vector<uint8_t> compressed(64);
  int64_t comp_size = 64;
  ASSERT_OK(PforWrapper<T>::Encode(nullptr, 0, compressed.data(), &comp_size));
  ASSERT_EQ(PforConstants::kHeaderSize, comp_size);

  ASSERT_OK_AND_ASSIGN(const int32_t count,
                       PforWrapper<T>::DecodeElementCount(compressed.data(), comp_size));
  ASSERT_EQ(0, count);
  ASSERT_OK(PforWrapper<T>::Decode(compressed.data(), comp_size, 0, nullptr));
}

TYPED_TEST(PforTest, EncodeRejectsNegativeCount) {
  using T = TypeParam;
  std::vector<T> values = {1, 2, 3};
  std::vector<uint8_t> compressed(64);
  int64_t comp_size = 64;
  ASSERT_RAISES(Invalid,
                PforWrapper<T>::Encode(values.data(), -1, compressed.data(), &comp_size));
}

TYPED_TEST(PforTest, EncodeRejectsNullInputWithValuesToEncode) {
  using T = TypeParam;
  std::vector<uint8_t> compressed(64);
  int64_t comp_size = 64;
  ASSERT_RAISES(Invalid,
                PforWrapper<T>::Encode(nullptr, 3, compressed.data(), &comp_size));
}

TYPED_TEST(PforTest, EncodeRejectsUnrepresentableVectorSize) {
  using T = TypeParam;
  std::vector<T> values = {1, 2, 3};
  std::vector<uint8_t> compressed(64);

  // Not a power of two.
  int64_t comp_size = 64;
  ASSERT_RAISES(Invalid, PforWrapper<T>::Encode(values.data(), 3, 1000, compressed.data(),
                                                &comp_size));

  // A power of two, but the header's log_vector_size field cannot describe it.
  comp_size = 64;
  ASSERT_RAISES(
      Invalid, PforWrapper<T>::Encode(values.data(), 3, PforConstants::kMaxVectorSize * 2,
                                      compressed.data(), &comp_size));
}

// ======================================================================
// Compression Ratio Test

TYPED_TEST(PforTest, ClusteredDataCompresses) {
  using T = TypeParam;
  // Data clustered around 1000 with one outlier
  auto values = TestFixture::RandomValues(1024, 1000, 1255, /*seed=*/42);
  values[500] = 999999;

  int64_t comp_size = 0;
  this->RoundTripPage(values, &comp_size);

  const size_t plain_size = 1024 * sizeof(T);
  EXPECT_LT(comp_size, static_cast<int64_t>(plain_size / 2));
}

// ======================================================================
// Corrupt Page Tests
//
// A PFOR page can come from anywhere, so Decode has to reject a header that
// disagrees with the buffer it arrived in rather than trusting it and reading
// or writing out of bounds.

namespace {

// A tight cluster plus one far outlier, so the cost model packs narrow and
// patches the outlier. The outlier is 31 bits out rather than 16: an exception
// costs a position plus a full-width value, so a 16-bit outlier is cheaper to
// pack than to patch once the values are 64 bits wide, and the tests below need
// a vector that actually has an exception section.
template <typename T>
std::vector<T> ClusterWithOutlier() {
  return {10, 20, 30, 40, 2'000'000'000};
}

}  // namespace

TYPED_TEST(PforTest, DecodeElementCountReadsThePageHeader) {
  using T = TypeParam;
  std::vector<T> values(2048);
  std::iota(values.begin(), values.end(), 0);
  auto compressed = TestFixture::EncodePage(values);

  ASSERT_OK_AND_ASSIGN(const int32_t count, PforWrapper<T>::DecodeElementCount(
                                                compressed.data(), compressed.size()));
  EXPECT_EQ(count, static_cast<int32_t>(values.size()));
}

TYPED_TEST(PforTest, DecodeElementCountRejectsMalformedInput) {
  using T = TypeParam;
  const std::vector<T> values = {10, 20, 30, 40, 50};
  auto compressed = TestFixture::EncodePage(values);

  ASSERT_RAISES(Invalid, PforWrapper<T>::DecodeElementCount(
                             compressed.data(), PforConstants::kHeaderSize - 1));
  ASSERT_RAISES(Invalid, PforWrapper<T>::DecodeElementCount(nullptr, compressed.size()));
  ASSERT_RAISES(Invalid, PforWrapper<T>::DecodeElementCount(compressed.data(), -1));
}

TYPED_TEST(PforTest, CorruptBufferTooSmallForHeader) {
  using T = TypeParam;
  const std::vector<T> values = {10, 20, 30, 40, 50};
  auto compressed = TestFixture::EncodePage(values);

  std::vector<T> decoded(values.size());
  ASSERT_RAISES(Invalid,
                PforWrapper<T>::Decode(compressed.data(), PforConstants::kHeaderSize - 1,
                                       5, decoded.data()));
}

TYPED_TEST(PforTest, CorruptElementCountExceedsOutputCapacity) {
  using T = TypeParam;
  const std::vector<T> values = {10, 20, 30, 40, 50};
  auto compressed = TestFixture::EncodePage(values);

  // num_elements lives at header byte 3.
  StoreLE32(compressed.data() + 3, 6);

  std::vector<T> decoded(values.size());
  ASSERT_RAISES(Invalid, PforWrapper<T>::Decode(compressed.data(), compressed.size(), 5,
                                                decoded.data()));
}

TYPED_TEST(PforTest, CorruptElementCountBelowOutputCapacity) {
  using T = TypeParam;
  const std::vector<T> values = {10, 20, 30, 40, 50};
  auto compressed = TestFixture::EncodePage(values);

  // A count short of the caller's capacity would fill part of the output and
  // leave the rest holding whatever it held before.
  StoreLE32(compressed.data() + 3, 4);

  std::vector<T> decoded(values.size());
  ASSERT_RAISES(Invalid, PforWrapper<T>::Decode(compressed.data(), compressed.size(), 5,
                                                decoded.data()));
}

TYPED_TEST(PforTest, CorruptOffsetArrayTruncated) {
  using T = TypeParam;
  // Four vectors, so the offset array is 16 bytes; hand Decode a buffer with
  // room for only two of them.
  std::vector<T> values(4096);
  std::iota(values.begin(), values.end(), 0);
  auto compressed = TestFixture::EncodePage(values);

  std::vector<T> decoded(values.size());
  ASSERT_RAISES(Invalid,
                PforWrapper<T>::Decode(compressed.data(), PforConstants::kHeaderSize + 8,
                                       4096, decoded.data()));
}

TYPED_TEST(PforTest, CorruptVectorOffsetPastEndOfBuffer) {
  using T = TypeParam;
  std::vector<T> values(2048);
  std::iota(values.begin(), values.end(), 0);
  auto compressed = TestFixture::EncodePage(values);

  // Point the second vector past the end of the buffer.
  StoreLE32(compressed.data() + TestFixture::kFirstVectorOffset,
            static_cast<uint32_t>(compressed.size()));

  std::vector<T> decoded(values.size());
  ASSERT_RAISES(Invalid, PforWrapper<T>::Decode(compressed.data(), compressed.size(),
                                                2048, decoded.data()));
}

TYPED_TEST(PforTest, CorruptFirstVectorOffsetSkipsTheOffsetArray) {
  using T = TypeParam;
  const std::vector<T> values = {10, 20, 30, 40, 50};
  auto compressed = TestFixture::EncodePage(values);

  // One vector, so its data starts right after the four byte offset array.
  StoreLE32(compressed.data() + PforConstants::kHeaderSize, 5);

  std::vector<T> decoded(values.size());
  ASSERT_RAISES(Invalid, PforWrapper<T>::Decode(compressed.data(), compressed.size(), 5,
                                                decoded.data()));
}

TYPED_TEST(PforTest, CorruptVectorOffsetsRunBackwards) {
  using T = TypeParam;
  std::vector<T> values(2048);
  std::iota(values.begin(), values.end(), 0);
  auto compressed = TestFixture::EncodePage(values);

  // Point the second vector at the first one's data. Both offsets are inside the
  // buffer, so only reading them as a chain catches it.
  StoreLE32(compressed.data() + TestFixture::kFirstVectorOffset,
            static_cast<uint32_t>(2 * sizeof(PforConstants::OffsetType)));

  std::vector<T> decoded(values.size());
  ASSERT_RAISES(Invalid, PforWrapper<T>::Decode(compressed.data(), compressed.size(),
                                                2048, decoded.data()));
}

TYPED_TEST(PforTest, CorruptExceptionPositionPastEndOfVector) {
  using T = TypeParam;
  const std::vector<T> values = ClusterWithOutlier<T>();
  auto compressed = TestFixture::EncodePage(values);

  // Locate the stored exception position: the vector info, then the packed
  // deltas at whatever width the cost model chose.
  const int64_t vector_start = TestFixture::kFirstVectorOffset;
  ASSERT_GT(compressed[vector_start + TestFixture::kNumExceptionsOffset], 0)
      << "the cost model packed this vector without exceptions, so there is no "
         "exception position to corrupt";
  const uint8_t bit_width = compressed[vector_start + TestFixture::kBitWidthOffset];
  const int64_t packed_bytes = (5 * bit_width + 7) / 8;
  const int64_t position_offset =
      vector_start + PforVectorInfo<T>::kStoredSize + packed_bytes;

  // Point the exception at index 100 of a five-element vector.
  compressed[position_offset] = 100;
  compressed[position_offset + 1] = 0;

  std::vector<T> decoded(5);
  ASSERT_RAISES(Invalid, PforWrapper<T>::Decode(compressed.data(), compressed.size(), 5,
                                                decoded.data()));
}

TYPED_TEST(PforTest, CorruptExceptionCountExceedsVectorLength) {
  using T = TypeParam;
  const std::vector<T> values = ClusterWithOutlier<T>();
  auto compressed = TestFixture::EncodePage(values);

  const int64_t vector_start = TestFixture::kFirstVectorOffset;
  compressed[vector_start + TestFixture::kNumExceptionsOffset] = 6;
  compressed[vector_start + TestFixture::kNumExceptionsOffset + 1] = 0;

  std::vector<T> decoded(5);
  ASSERT_RAISES(Invalid, PforWrapper<T>::Decode(compressed.data(), compressed.size(), 5,
                                                decoded.data()));
}

TYPED_TEST(PforTest, CorruptExceptionDataTruncated) {
  using T = TypeParam;
  const std::vector<T> values = ClusterWithOutlier<T>();
  auto compressed = TestFixture::EncodePage(values);

  // Drop the trailing exception value, leaving its position behind.
  ASSERT_GT(
      compressed[TestFixture::kFirstVectorOffset + TestFixture::kNumExceptionsOffset], 0);
  std::vector<T> decoded(5);
  ASSERT_RAISES(
      Invalid, PforWrapper<T>::Decode(compressed.data(),
                                      static_cast<int64_t>(compressed.size() - sizeof(T)),
                                      5, decoded.data()));
}

// ======================================================================
// Output sizing

TYPED_TEST(PforTest, MaxCompressedSizeRejectsInvalidArguments) {
  using T = TypeParam;
  ASSERT_RAISES(Invalid, PforWrapper<T>::GetMaxCompressedSize(-1));
  // A zero vector_size would divide by zero on the way to the answer.
  for (const int32_t vector_size : {0, 3, 1000, PforConstants::kMaxVectorSize * 2}) {
    SCOPED_TRACE("vector_size=" + std::to_string(vector_size));
    ASSERT_RAISES(Invalid, PforWrapper<T>::GetMaxCompressedSize(1024, vector_size));
    std::vector<T> values(16, 0);
    int64_t comp_size = 1 << 20;
    std::vector<uint8_t> compressed(comp_size);
    ASSERT_RAISES(Invalid, PforWrapper<T>::Encode(values.data(), 16, vector_size,
                                                  compressed.data(), &comp_size));
  }
}

TYPED_TEST(PforTest, EncodeRejectsUndersizedOutputBuffer) {
  using T = TypeParam;
  std::vector<T> values(1024);
  std::iota(values.begin(), values.end(), 0);
  ASSERT_OK_AND_ASSIGN(const int64_t max_size,
                       PforWrapper<T>::GetMaxCompressedSize(1024));

  std::vector<uint8_t> compressed(max_size);
  int64_t comp_size = max_size - 1;
  ASSERT_RAISES(Invalid, PforWrapper<T>::Encode(values.data(), 1024, compressed.data(),
                                                &comp_size));
}

TYPED_TEST(PforTest, SerializeVectorRejectsUndersizedDestination) {
  using T = TypeParam;
  std::vector<T> values(64);
  std::iota(values.begin(), values.end(), 0);
  const auto encoded = PforCompression<T>::EncodeVector(values.data(), 64);
  const int64_t needed = PforCompression<T>::SerializedVectorSize(encoded, 64);

  std::vector<uint8_t> buffer(needed);
  ASSERT_OK_AND_ASSIGN(const int64_t written,
                       PforCompression<T>::SerializeVector(encoded, 64, buffer));
  ASSERT_EQ(needed, written);
  ASSERT_RAISES(Invalid, PforCompression<T>::SerializeVector(
                             encoded, 64, std::span<uint8_t>(buffer).subspan(1)));
}

// SerializedVectorSize is derived from the vector info, while the copy that
// follows it takes its lengths from the sections, so a vector whose info and
// sections disagree has to be refused rather than written past the size the
// caller reserved.
TYPED_TEST(PforTest, SerializeVectorRejectsInfoInconsistentWithSections) {
  using T = TypeParam;
  std::vector<T> values(64);
  std::iota(values.begin(), values.end(), 0);
  const auto encoded = PforCompression<T>::EncodeVector(values.data(), 64);
  std::vector<uint8_t> buffer(PforCompression<T>::SerializedVectorSize(encoded, 64) + 64);

  {
    // A bit width that does not account for the packed bytes present.
    auto corrupted = encoded;
    corrupted.mutable_info().set_bit_width(
        static_cast<uint8_t>(encoded.info().bit_width() + 1));
    ASSERT_RAISES(Invalid, PforCompression<T>::SerializeVector(corrupted, 64, buffer));
  }
  {
    // An exception count with no exception sections behind it.
    auto corrupted = encoded;
    corrupted.mutable_info().set_num_exceptions(1);
    ASSERT_RAISES(Invalid, PforCompression<T>::SerializeVector(corrupted, 64, buffer));
  }
}

// GetMaxCompressedSize is what the encoder allocates, and it is derived from the
// cost model rather than from a worst case that assumes every value is an
// exception, so inputs that push the model in different directions all have to
// land inside it.
TYPED_TEST(PforTest, MaxCompressedSizeBoundHoldsForAdversarialInputs) {
  using T = TypeParam;
  constexpr int32_t kNumValues = 3000;
  std::mt19937 rng(7);
  std::uniform_int_distribution<T> full_range(std::numeric_limits<T>::min(),
                                              std::numeric_limits<T>::max());

  std::vector<std::vector<T>> inputs;
  // Incompressible: every delta needs the full width, so the model picks it.
  inputs.emplace_back(kNumValues);
  for (auto& v : inputs.back()) v = full_range(rng);
  // A narrow cluster with a scatter of far-away values, which is the case that
  // trades bit width against exception count.
  inputs.emplace_back(kNumValues, 1000);
  for (int32_t i = 0; i < kNumValues; i += 3) {
    inputs.back()[i] = full_range(rng);
  }
  // Alternating extremes, so the frame of reference cannot help.
  inputs.emplace_back(kNumValues);
  for (int32_t i = 0; i < kNumValues; ++i) {
    inputs.back()[i] =
        (i % 2 == 0) ? std::numeric_limits<T>::min() : std::numeric_limits<T>::max();
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    SCOPED_TRACE("input " + std::to_string(i));
    this->RoundTripPage(inputs[i]);
  }
}

// ======================================================================
// Interleaved bit-packing layout
//
// The interleaved layout writes the same number of bytes as the sequential one,
// so a wrong-layout decode returns wrong values rather than running out of
// input. That makes these round trips the only thing standing between the two
// layouts, and it is why the tests below also assert that the two payloads are
// genuinely different bytes: without that, a silent fallback on both sides
// would pass every round trip here.

namespace {

/// The packing_mode byte a page recorded, which is the first byte of the header.
uint8_t PackingModeOf(const std::vector<uint8_t>& page) { return page[0]; }

/// Build a vector whose deltas need exactly `width` bits, so that a loop over
/// widths reaches every one of the packing kernels.
///
/// A frame of reference of zero exercises the plain unpack; a non-zero one the
/// unpack that folds the frame into its own store. At the full width of the type
/// there is no choice between them: deltas that wide only fit under the type's
/// minimum, so that width is always framed.
template <typename T>
std::vector<T> ValuesForWidth(int width, bool zero_frame, int32_t num_values,
                              uint32_t seed) {
  using UnsignedT = typename PforTypeTraits<T>::UnsignedType;
  constexpr int kBits = 8 * static_cast<int>(sizeof(T));

  const T base =
      width == kBits ? std::numeric_limits<T>::min() : (zero_frame ? T{0} : T{-1000});
  std::vector<T> values(num_values, base);
  if (width == 0) {
    return values;
  }

  const UnsignedT mask = width == kBits
                             ? ~UnsignedT{0}
                             : static_cast<UnsignedT>((UnsignedT{1} << width) - 1);
  std::mt19937_64 rng(seed);
  for (int32_t i = 0; i < num_values; ++i) {
    values[i] = static_cast<T>(static_cast<UnsignedT>(base) +
                               (static_cast<UnsignedT>(rng()) & mask));
  }
  // Pin the two ends, so the vector cannot land on a narrower width by chance.
  values[0] = base;
  values[1] = static_cast<T>(static_cast<UnsignedT>(base) + mask);
  return values;
}

}  // namespace

/// Every bit width round-trips through the interleaved layout, with and without
/// a frame of reference to fold into the unpack.
///
/// For int64 this exercises the other side of the same request: the layout does
/// not apply, so the page must come back correct in the sequential layout.
TYPED_TEST(PforTest, InterleavedAllBitWidths) {
  using T = TypeParam;
  constexpr auto kInterleaved = PackingMode::kForBitPackInterleaved;
  const auto num_values = static_cast<int32_t>(PforConstants::kPforVectorSize);

  for (int width = 0; width <= 32; ++width) {
    for (const bool zero_frame : {true, false}) {
      // A 32-bit delta only fits under the minimum of a 32-bit type, so there is
      // no unframed form of that width to test.
      if (zero_frame && width == 8 * static_cast<int>(sizeof(T))) continue;
      const auto values =
          ValuesForWidth<T>(width, zero_frame, num_values, static_cast<uint32_t>(width));

      int64_t comp_size = PforWrapper<T>::GetMaxCompressedSize(num_values).ValueOrDie();
      std::vector<uint8_t> page(comp_size);
      ASSERT_OK(PforWrapper<T>::Encode(values.data(), num_values, page.data(), &comp_size,
                                       kInterleaved));
      page.resize(comp_size);

      std::vector<T> decoded(num_values);
      ASSERT_OK(
          PforWrapper<T>::Decode(page.data(), comp_size, num_values, decoded.data()))
          << "width " << width << " zero_frame " << zero_frame;
      ASSERT_EQ(values, decoded) << "width " << width << " zero_frame " << zero_frame;

      const uint8_t expected_mode =
          static_cast<uint8_t>(sizeof(T) == 4 ? kInterleaved : PackingMode::kForBitPack);
      ASSERT_EQ(expected_mode, PackingModeOf(page)) << "width " << width;
    }
  }
}

/// Exceptions are stored at the positions the encoder recorded, and the
/// interleaved layout does not move values between positions, so those
/// positions need no translation.
TYPED_TEST(PforTest, InterleavedWithExceptions) {
  using T = TypeParam;
  const auto num_values = static_cast<int32_t>(PforConstants::kPforVectorSize);
  auto values = TestFixture::RandomValues(num_values, T{100}, T{140}, /*seed=*/7);
  // First, middle and last position, so an off-by-one in either direction shows.
  values[0] = std::numeric_limits<T>::max() / 2;
  values[num_values / 2] = std::numeric_limits<T>::max() / 2;
  values[num_values - 1] = std::numeric_limits<T>::max() / 2;

  int64_t comp_size = PforWrapper<T>::GetMaxCompressedSize(num_values).ValueOrDie();
  std::vector<uint8_t> page(comp_size);
  ASSERT_OK(PforWrapper<T>::Encode(values.data(), num_values, page.data(), &comp_size,
                                   PackingMode::kForBitPackInterleaved));

  std::vector<T> decoded(num_values);
  ASSERT_OK(PforWrapper<T>::Decode(page.data(), comp_size, num_values, decoded.data()));
  ASSERT_EQ(values, decoded);
}

/// A page whose length is not a multiple of the vector size ends in a short
/// vector, which the interleaved layout cannot hold. Both sides derive that from
/// the element count, so it round-trips without a per-vector flag.
TYPED_TEST(PforTest, InterleavedShortTailVector) {
  using T = TypeParam;
  for (const int32_t num_values : {1, 1023, 1025, 2048, 3000, 4096}) {
    auto values = TestFixture::RandomValues(num_values, T{0}, T{5000},
                                            static_cast<uint32_t>(num_values));
    int64_t comp_size = PforWrapper<T>::GetMaxCompressedSize(num_values).ValueOrDie();
    std::vector<uint8_t> page(comp_size);
    ASSERT_OK(PforWrapper<T>::Encode(values.data(), num_values, page.data(), &comp_size,
                                     PackingMode::kForBitPackInterleaved));

    std::vector<T> decoded(num_values);
    ASSERT_OK(PforWrapper<T>::Decode(page.data(), comp_size, num_values, decoded.data()))
        << "num_values " << num_values;
    ASSERT_EQ(values, decoded) << "num_values " << num_values;
  }
}

/// The two layouts write the same number of bytes, so choosing one costs no
/// space.
TYPED_TEST(PforTest, InterleavedCostsNoSpace) {
  using T = TypeParam;
  for (const int32_t num_values : {1024, 2048, 3000}) {
    auto values = TestFixture::RandomValues(num_values, T{0}, T{5000}, /*seed=*/11);

    int64_t sequential_size =
        PforWrapper<T>::GetMaxCompressedSize(num_values).ValueOrDie();
    std::vector<uint8_t> sequential(sequential_size);
    ASSERT_OK(PforWrapper<T>::Encode(values.data(), num_values, sequential.data(),
                                     &sequential_size));

    int64_t interleaved_size =
        PforWrapper<T>::GetMaxCompressedSize(num_values).ValueOrDie();
    std::vector<uint8_t> interleaved(interleaved_size);
    ASSERT_OK(PforWrapper<T>::Encode(values.data(), num_values, interleaved.data(),
                                     &interleaved_size,
                                     PackingMode::kForBitPackInterleaved));

    ASSERT_EQ(sequential_size, interleaved_size) << "num_values " << num_values;
  }
}

/// The two layouts really are different bytes, which is what keeps the round
/// trips above honest: if a mode request quietly fell back to sequential on both
/// the encode and the decode side, every one of them would still pass.
///
/// At 32 bits the layouts coincide -- the sequential layout puts value i in word
/// i, and so does the interleaved one -- so the payloads are identical there by
/// construction, not by accident.
TEST(PforInterleavedTest, LayoutsDifferBelowThirtyTwoBits) {
  using T = int32_t;
  const auto num_values = static_cast<int32_t>(PforConstants::kPforVectorSize);

  for (int width = 1; width <= 32; ++width) {
    const auto values = ValuesForWidth<T>(width, /*zero_frame=*/width < 32, num_values,
                                          static_cast<uint32_t>(width));

    int64_t sequential_size =
        PforWrapper<T>::GetMaxCompressedSize(num_values).ValueOrDie();
    std::vector<uint8_t> sequential(sequential_size);
    ASSERT_OK(PforWrapper<T>::Encode(values.data(), num_values, sequential.data(),
                                     &sequential_size));
    sequential.resize(sequential_size);

    int64_t interleaved_size =
        PforWrapper<T>::GetMaxCompressedSize(num_values).ValueOrDie();
    std::vector<uint8_t> interleaved(interleaved_size);
    ASSERT_OK(PforWrapper<T>::Encode(values.data(), num_values, interleaved.data(),
                                     &interleaved_size,
                                     PackingMode::kForBitPackInterleaved));
    interleaved.resize(interleaved_size);

    ASSERT_EQ(sequential.size(), interleaved.size()) << "width " << width;

    // The cost model is free to choose a narrower width and pay for exceptions,
    // which would leave this loop testing one width many times over. Read the
    // width it settled on out of the vector's own metadata: header, then the
    // one-entry offset array, then the frame of reference.
    const int64_t bit_width_offset =
        PforConstants::kHeaderSize +
        static_cast<int64_t>(sizeof(PforConstants::OffsetType)) +
        static_cast<int64_t>(sizeof(T));
    ASSERT_EQ(width, sequential[bit_width_offset]) << "requested width " << width;

    // Skip the header, whose packing_mode byte differs on purpose.
    const bool payload_differs =
        !std::equal(sequential.begin() + PforConstants::kHeaderSize, sequential.end(),
                    interleaved.begin() + PforConstants::kHeaderSize);
    if (width == 32) {
      ASSERT_FALSE(payload_differs) << "the layouts coincide at 32 bits";
    } else {
      ASSERT_TRUE(payload_differs) << "width " << width;
    }
  }
}

/// A request the page cannot satisfy is recorded as the sequential layout rather
/// than refused, so one writer-side setting can cover a file whose columns are
/// not all eligible.
TEST(PforInterleavedTest, UnsatisfiableRequestRecordsSequential) {
  const auto num_values = static_cast<int32_t>(PforConstants::kPforVectorSize);
  constexpr auto kInterleaved = PackingMode::kForBitPackInterleaved;

  // Eight bytes per element.
  {
    std::vector<int64_t> values(num_values);
    std::iota(values.begin(), values.end(), int64_t{1} << 40);
    int64_t comp_size =
        PforWrapper<int64_t>::GetMaxCompressedSize(num_values).ValueOrDie();
    std::vector<uint8_t> page(comp_size);
    ASSERT_OK(PforWrapper<int64_t>::Encode(values.data(), num_values, page.data(),
                                           &comp_size, kInterleaved));
    ASSERT_EQ(static_cast<uint8_t>(PackingMode::kForBitPack), PackingModeOf(page));

    std::vector<int64_t> decoded(num_values);
    ASSERT_OK(
        PforWrapper<int64_t>::Decode(page.data(), comp_size, num_values, decoded.data()));
    ASSERT_EQ(values, decoded);
  }

  // A vector size other than 1024.
  {
    std::vector<int32_t> values(num_values);
    std::iota(values.begin(), values.end(), 0);
    constexpr int32_t kVectorSize = 512;
    int64_t comp_size =
        PforWrapper<int32_t>::GetMaxCompressedSize(num_values, kVectorSize).ValueOrDie();
    std::vector<uint8_t> page(comp_size);
    ASSERT_OK(PforWrapper<int32_t>::Encode(values.data(), num_values, kVectorSize,
                                           page.data(), &comp_size, kInterleaved));
    ASSERT_EQ(static_cast<uint8_t>(PackingMode::kForBitPack), PackingModeOf(page));

    std::vector<int32_t> decoded(num_values);
    ASSERT_OK(
        PforWrapper<int32_t>::Decode(page.data(), comp_size, num_values, decoded.data()));
    ASSERT_EQ(values, decoded);
  }
}

/// A page that declares the interleaved layout alongside a vector size or an
/// element width that cannot support it is rejected, rather than decoded into
/// wrong values.
TEST(PforInterleavedTest, ForgedHeaderIsRejected) {
  const auto num_values = static_cast<int32_t>(PforConstants::kPforVectorSize);
  std::vector<int32_t> values(num_values);
  std::iota(values.begin(), values.end(), 0);
  constexpr int32_t kVectorSize = 512;

  int64_t comp_size =
      PforWrapper<int32_t>::GetMaxCompressedSize(num_values, kVectorSize).ValueOrDie();
  std::vector<uint8_t> page(comp_size);
  ASSERT_OK(PforWrapper<int32_t>::Encode(values.data(), num_values, kVectorSize,
                                         page.data(), &comp_size));

  std::vector<int32_t> decoded(num_values);
  page[0] = static_cast<uint8_t>(PackingMode::kForBitPackInterleaved);
  ASSERT_RAISES(Invalid, PforWrapper<int32_t>::Decode(page.data(), comp_size, num_values,
                                                      decoded.data()));

  // An unknown layout is an error too, so a page written by a later encoder is
  // refused instead of misread. The two layouts are the same size, so a reader
  // that guessed would not run short.
  page[0] = 2;
  ASSERT_RAISES(Invalid, PforWrapper<int32_t>::Decode(page.data(), comp_size, num_values,
                                                      decoded.data()));
}

}  // namespace arrow::util::pfor
