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

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "arrow/testing/gtest_util.h"
#include "arrow/util/alp/alp.h"
#include "arrow/util/alp/alp_constants.h"
#include "arrow/util/alp/alp_rd.h"
#include "arrow/util/alp/alp_sampler.h"
#include "arrow/util/alp/alp_wrapper.h"

namespace arrow {
namespace util {
namespace alp {

// ============================================================================
// ALP-RD Constants Tests
// ============================================================================

TEST(AlpRdConstantsTest, BasicConstants) {
  EXPECT_EQ(AlpRdConstants::kAlpVectorSize, 1024);
  EXPECT_EQ(AlpRdConstants::kCuttingLimit, 16);
  EXPECT_EQ(AlpRdConstants::kDefaultMaxDictionaryBitWidth, 8);
}

TEST(AlpRdConstantsTest, GetMaxDictionarySize) {
  EXPECT_EQ(AlpRdConstants::GetMaxDictionarySize(1), 2);
  EXPECT_EQ(AlpRdConstants::GetMaxDictionarySize(4), 16);
  EXPECT_EQ(AlpRdConstants::GetMaxDictionarySize(8), 256);
}

// ============================================================================
// AlpRdEncodingSettings Tests
// ============================================================================

TEST(AlpRdEncodingSettingsTest, DefaultConstruction) {
  AlpRdEncodingSettings settings;
  EXPECT_EQ(settings.left_bit_width, 0);
  EXPECT_EQ(settings.right_bit_width, 0);
  EXPECT_EQ(settings.num_dict_entries, 0);
  EXPECT_TRUE(settings.flat_dictionary.empty());
}

TEST(AlpRdEncodingSettingsTest, ConstructionWithBitWidth) {
  AlpRdEncodingSettings settings(8);
  EXPECT_EQ(settings.flat_dictionary.size(), 257);  // 2^8 + 1
}

TEST(AlpRdEncodingSettingsTest, StoreLoad) {
  AlpRdEncodingSettings original(AlpRdConstants::kDefaultMaxDictionaryBitWidth);
  original.left_bit_width = 5;
  original.right_bit_width = 48;
  original.num_dict_entries = 32;
  for (uint16_t i = 0; i < original.num_dict_entries; ++i) {
    original.flat_dictionary[i] = i * 100;
  }

  // Store
  const uint64_t stored_size = original.GetStoredSize();
  std::vector<char> buffer(stored_size);
  original.Store({buffer.data(), buffer.size()});

  // Load
  AlpRdEncodingSettings loaded = AlpRdEncodingSettings::Load({buffer.data(), buffer.size()});

  // Verify
  EXPECT_EQ(loaded.left_bit_width, original.left_bit_width);
  EXPECT_EQ(loaded.right_bit_width, original.right_bit_width);
  EXPECT_EQ(loaded.num_dict_entries, original.num_dict_entries);
  for (uint16_t i = 0; i < original.num_dict_entries; ++i) {
    EXPECT_EQ(loaded.flat_dictionary[i], original.flat_dictionary[i]);
  }
}

TEST(AlpRdEncodingSettingsTest, Equality) {
  AlpRdEncodingSettings s1(8);
  s1.left_bit_width = 4;
  s1.right_bit_width = 48;
  s1.num_dict_entries = 16;
  for (uint16_t i = 0; i < s1.num_dict_entries; ++i) {
    s1.flat_dictionary[i] = i;
  }

  AlpRdEncodingSettings s2(8);
  s2.left_bit_width = 4;
  s2.right_bit_width = 48;
  s2.num_dict_entries = 16;
  for (uint16_t i = 0; i < s2.num_dict_entries; ++i) {
    s2.flat_dictionary[i] = i;
  }

  EXPECT_TRUE(s1 == s2);

  s2.left_bit_width = 5;
  EXPECT_FALSE(s1 == s2);
}

// ============================================================================
// AlpRdVectorMetadata Tests
// ============================================================================

TEST(AlpRdVectorMetadataTest, StoredSize) {
  EXPECT_EQ(AlpRdVectorMetadata::GetStoredSize(), 20);
}

TEST(AlpRdVectorMetadataTest, StoreLoad) {
  AlpRdVectorMetadata original;
  original.left_bit_width = 6;
  original.right_bit_width = 52;
  original.num_exceptions = 5;
  original.left_bit_packed_size = 768;
  original.right_bit_packed_size = 6656;

  // Store
  std::vector<char> buffer(AlpRdVectorMetadata::GetStoredSize());
  original.Store({buffer.data(), buffer.size()});

  // Load
  AlpRdVectorMetadata loaded = AlpRdVectorMetadata::Load({buffer.data(), buffer.size()});

  // Verify
  EXPECT_EQ(loaded, original);
}

TEST(AlpRdVectorMetadataTest, GetDataSize) {
  AlpRdVectorMetadata meta;
  meta.left_bit_packed_size = 100;
  meta.right_bit_packed_size = 200;
  meta.num_exceptions = 5;

  // Data size = left + right + positions (2B each) + exceptions (2B each)
  const uint64_t expected = 100 + 200 + 5 * 2 + 5 * 2;
  EXPECT_EQ(meta.GetDataSize(), expected);
}

// ============================================================================
// AlpRdCompression Tests (Float)
// ============================================================================

class AlpRdCompressionFloatTest : public ::testing::Test {
 protected:
  using ExactType = typename AlpTypedConstants<float>::FloatingToExact;

  void TestCompressDecompressFloat(const std::vector<float>& input) {
    // Convert to exact representation
    std::vector<ExactType> exact_input(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
      std::memcpy(&exact_input[i], &input[i], sizeof(float));
    }

    // Create preset
    AlpRdEncodingPreset preset =
        AlpRdCompression<float>::CreateEncodingPreset(exact_input);

    // Compress
    auto encoded = AlpRdCompression<float>::CompressVector(
        exact_input.data(), exact_input.size(), preset);

    // Decompress
    std::vector<float> output(input.size());
    AlpRdCompression<float>::DecompressVector(encoded, output.data(),
                                              static_cast<uint16_t>(input.size()));

    // Verify lossless
    for (size_t i = 0; i < input.size(); ++i) {
      EXPECT_EQ(std::memcmp(&output[i], &input[i], sizeof(float)), 0)
          << "Bit mismatch at index " << i;
    }
  }
};

TEST_F(AlpRdCompressionFloatTest, RandomFloats) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

  std::vector<float> input(64);
  for (auto& v : input) {
    v = dist(rng);
  }

  TestCompressDecompressFloat(input);
}

TEST_F(AlpRdCompressionFloatTest, SmallVector) {
  std::vector<float> input = {1.5f, 2.7f, 3.14159f, 4.0f, 5.5f};
  TestCompressDecompressFloat(input);
}

TEST_F(AlpRdCompressionFloatTest, SimilarValues) {
  // Values with similar high bits should compress well with ALP-RD
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(1000000.0f, 1000001.0f);

  std::vector<float> input(64);
  for (auto& v : input) {
    v = dist(rng);
  }

  TestCompressDecompressFloat(input);
}

TEST_F(AlpRdCompressionFloatTest, LargeVector) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, 100.0f);

  std::vector<float> input(1024);  // Full vector size
  for (auto& v : input) {
    v = dist(rng);
  }

  TestCompressDecompressFloat(input);
}

// ============================================================================
// AlpRdCompression Tests (Double)
// ============================================================================

class AlpRdCompressionDoubleTest : public ::testing::Test {
 protected:
  using ExactType = typename AlpTypedConstants<double>::FloatingToExact;

  void TestCompressDecompressDouble(const std::vector<double>& input) {
    // Convert to exact representation
    std::vector<ExactType> exact_input(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
      std::memcpy(&exact_input[i], &input[i], sizeof(double));
    }

    // Create preset
    AlpRdEncodingPreset preset =
        AlpRdCompression<double>::CreateEncodingPreset(exact_input);

    // Compress
    auto encoded = AlpRdCompression<double>::CompressVector(
        exact_input.data(), exact_input.size(), preset);

    // Decompress
    std::vector<double> output(input.size());
    AlpRdCompression<double>::DecompressVector(encoded, output.data(),
                                               static_cast<uint16_t>(input.size()));

    // Verify lossless
    for (size_t i = 0; i < input.size(); ++i) {
      EXPECT_EQ(std::memcmp(&output[i], &input[i], sizeof(double)), 0)
          << "Bit mismatch at index " << i;
    }
  }
};

TEST_F(AlpRdCompressionDoubleTest, RandomDoubles) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

  std::vector<double> input(64);
  for (auto& v : input) {
    v = dist(rng);
  }

  TestCompressDecompressDouble(input);
}

TEST_F(AlpRdCompressionDoubleTest, SmallVector) {
  std::vector<double> input = {1.5, 2.7, 3.14159265358979, 4.0, 5.5};
  TestCompressDecompressDouble(input);
}

TEST_F(AlpRdCompressionDoubleTest, FullVector) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(0.0, 100.0);

  std::vector<double> input(1024);  // Full vector size
  for (auto& v : input) {
    v = dist(rng);
  }

  TestCompressDecompressDouble(input);
}

// ============================================================================
// AlpRdEncodedVector Store/Load Tests
// ============================================================================

template <typename T>
class AlpRdEncodedVectorTest : public ::testing::Test {};

using FloatingTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(AlpRdEncodedVectorTest, FloatingTypes);

TYPED_TEST(AlpRdEncodedVectorTest, StoreLoadRoundTrip) {
  using ExactType = typename AlpTypedConstants<TypeParam>::FloatingToExact;

  // Create test data
  std::mt19937 rng(42);
  std::uniform_real_distribution<TypeParam> dist(0.0, 100.0);

  std::vector<TypeParam> input(64);
  for (auto& v : input) {
    v = dist(rng);
  }

  std::vector<ExactType> exact_input(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    std::memcpy(&exact_input[i], &input[i], sizeof(TypeParam));
  }

  // Create preset and compress
  AlpRdEncodingPreset preset =
      AlpRdCompression<TypeParam>::CreateEncodingPreset(exact_input);
  auto encoded = AlpRdCompression<TypeParam>::CompressVector(
      exact_input.data(), exact_input.size(), preset);

  // Store
  const uint64_t stored_size = encoded.GetStoredSize();
  std::vector<char> buffer(stored_size);
  encoded.Store({buffer.data(), buffer.size()});

  // Load
  auto loaded = AlpRdEncodedVector<TypeParam>::Load(
      {buffer.data(), buffer.size()}, static_cast<uint16_t>(input.size()));

  // Decompress loaded
  std::vector<TypeParam> output(input.size());
  AlpRdCompression<TypeParam>::DecompressVector(loaded, output.data(),
                                                static_cast<uint16_t>(input.size()));

  // Verify
  for (size_t i = 0; i < input.size(); ++i) {
    EXPECT_EQ(std::memcmp(&output[i], &input[i], sizeof(TypeParam)), 0)
        << "Bit mismatch at index " << i;
  }
}

// ============================================================================
// AlpRdEncodedVectorView Tests
// ============================================================================

TYPED_TEST(AlpRdEncodedVectorTest, ViewDecompression) {
  using ExactType = typename AlpTypedConstants<TypeParam>::FloatingToExact;

  // Create test data
  std::mt19937 rng(42);
  std::uniform_real_distribution<TypeParam> dist(0.0, 100.0);

  std::vector<TypeParam> input(64);
  for (auto& v : input) {
    v = dist(rng);
  }

  std::vector<ExactType> exact_input(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    std::memcpy(&exact_input[i], &input[i], sizeof(TypeParam));
  }

  // Create preset and compress
  AlpRdEncodingPreset preset =
      AlpRdCompression<TypeParam>::CreateEncodingPreset(exact_input);
  auto encoded = AlpRdCompression<TypeParam>::CompressVector(
      exact_input.data(), exact_input.size(), preset);

  // Store
  std::vector<char> buffer(encoded.GetStoredSize());
  encoded.Store({buffer.data(), buffer.size()});

  // Load as view
  auto view = AlpRdEncodedVectorView<TypeParam>::LoadView(
      {buffer.data(), buffer.size()}, static_cast<uint16_t>(input.size()));

  // Decompress using view
  std::vector<TypeParam> output(input.size());
  AlpRdCompression<TypeParam>::DecompressVectorView(view, output.data(),
                                                    static_cast<uint16_t>(input.size()));

  // Verify
  for (size_t i = 0; i < input.size(); ++i) {
    EXPECT_EQ(std::memcmp(&output[i], &input[i], sizeof(TypeParam)), 0)
        << "Bit mismatch at index " << i;
  }
}

// ============================================================================
// AlpSampler with ALP-RD Tests
// ============================================================================

TEST(AlpSamplerTest, RecommendationForRandomData) {
  // Random data should likely recommend ALP-RD over ALP
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

  std::vector<double> input(4096);
  for (auto& v : input) {
    v = dist(rng);
  }

  AlpSampler<double> sampler;
  sampler.AddSample({input.data(), input.size()});
  auto result = sampler.Finalize(/*always_generate_alp_rd_preset=*/true);

  // Should have both presets
  EXPECT_TRUE(result.alp_rd_preset.has_value());

  // Check that the recommendation is valid
  EXPECT_TRUE(result.recommendation == AlpMode::kAlp ||
              result.recommendation == AlpMode::kAlpRd);
}

TEST(AlpSamplerTest, RecommendationForDecimalData) {
  // Decimal-like data should recommend ALP over ALP-RD
  std::vector<double> input(4096);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<double>(i) * 0.01;  // 0.00, 0.01, 0.02, ...
  }

  AlpSampler<double> sampler;
  sampler.AddSample({input.data(), input.size()});
  auto result = sampler.Finalize();

  // Decimal data should favor ALP
  EXPECT_EQ(result.recommendation, AlpMode::kAlp);
}

// ============================================================================
// AlpWrapper ALP-RD Integration Tests
// ============================================================================

template <typename T>
class AlpWrapperAlpRdTest : public ::testing::Test {};

TYPED_TEST_SUITE(AlpWrapperAlpRdTest, FloatingTypes);

TYPED_TEST(AlpWrapperAlpRdTest, EnforceAlpRdMode) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<TypeParam> dist(0.0, 100.0);

  std::vector<TypeParam> input(256);
  for (auto& v : input) {
    v = dist(rng);
  }

  const size_t max_compressed_size =
      AlpWrapper<TypeParam>::GetMaxCompressedSize(input.size() * sizeof(TypeParam));
  std::vector<char> compressed(max_compressed_size);
  size_t comp_size = compressed.size();

  // Encode with enforced ALP-RD mode
  AlpWrapper<TypeParam>::Encode(input.data(), input.size() * sizeof(TypeParam),
                                compressed.data(), &comp_size, AlpMode::kAlpRd);

  EXPECT_GT(comp_size, 0);

  // Decode
  std::vector<TypeParam> output(input.size());
  AlpWrapper<TypeParam>::Decode(output.data(), static_cast<uint32_t>(input.size()),
                                compressed.data(), comp_size);

  // Verify lossless
  for (size_t i = 0; i < input.size(); ++i) {
    EXPECT_EQ(std::memcmp(&output[i], &input[i], sizeof(TypeParam)), 0)
        << "Bit mismatch at index " << i;
  }
}

TYPED_TEST(AlpWrapperAlpRdTest, AutoModeSelection) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<TypeParam> dist(-1000.0, 1000.0);

  std::vector<TypeParam> input(1024);
  for (auto& v : input) {
    v = dist(rng);
  }

  const size_t max_compressed_size =
      AlpWrapper<TypeParam>::GetMaxCompressedSize(input.size() * sizeof(TypeParam));
  std::vector<char> compressed(max_compressed_size);
  size_t comp_size = compressed.size();

  // Encode with auto mode selection
  AlpWrapper<TypeParam>::Encode(input.data(), input.size() * sizeof(TypeParam),
                                compressed.data(), &comp_size);

  EXPECT_GT(comp_size, 0);

  // Decode
  std::vector<TypeParam> output(input.size());
  AlpWrapper<TypeParam>::Decode(output.data(), static_cast<uint32_t>(input.size()),
                                compressed.data(), comp_size);

  // Verify lossless
  for (size_t i = 0; i < input.size(); ++i) {
    EXPECT_EQ(std::memcmp(&output[i], &input[i], sizeof(TypeParam)), 0)
        << "Bit mismatch at index " << i;
  }
}

TYPED_TEST(AlpWrapperAlpRdTest, MultipleVectors) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<TypeParam> dist(0.0, 100.0);

  // Multiple vectors (more than kAlpVectorSize)
  std::vector<TypeParam> input(3000);
  for (auto& v : input) {
    v = dist(rng);
  }

  const size_t max_compressed_size =
      AlpWrapper<TypeParam>::GetMaxCompressedSize(input.size() * sizeof(TypeParam));
  std::vector<char> compressed(max_compressed_size);
  size_t comp_size = compressed.size();

  // Encode with enforced ALP-RD mode
  AlpWrapper<TypeParam>::Encode(input.data(), input.size() * sizeof(TypeParam),
                                compressed.data(), &comp_size, AlpMode::kAlpRd);

  EXPECT_GT(comp_size, 0);

  // Decode
  std::vector<TypeParam> output(input.size());
  AlpWrapper<TypeParam>::Decode(output.data(), static_cast<uint32_t>(input.size()),
                                compressed.data(), comp_size);

  // Verify lossless
  for (size_t i = 0; i < input.size(); ++i) {
    EXPECT_EQ(std::memcmp(&output[i], &input[i], sizeof(TypeParam)), 0)
        << "Bit mismatch at index " << i;
  }
}

// ============================================================================
// AlpRdMetadataCache Tests
// ============================================================================

TYPED_TEST(AlpRdEncodedVectorTest, MetadataCacheBasic) {
  using ExactType = typename AlpTypedConstants<TypeParam>::FloatingToExact;

  // Create test data for multiple vectors
  std::mt19937 rng(42);
  std::uniform_real_distribution<TypeParam> dist(0.0, 100.0);

  const uint32_t num_vectors = 3;
  const uint32_t vector_size = 64;
  const uint64_t total_elements = num_vectors * vector_size;

  std::vector<TypeParam> input(total_elements);
  for (auto& v : input) {
    v = dist(rng);
  }

  std::vector<ExactType> exact_input(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    std::memcpy(&exact_input[i], &input[i], sizeof(TypeParam));
  }

  // Create preset
  AlpRdEncodingPreset preset =
      AlpRdCompression<TypeParam>::CreateEncodingPreset(exact_input);

  // Compress all vectors and collect metadata
  std::vector<char> metadata_buffer(num_vectors * AlpRdVectorMetadata::GetStoredSize());
  uint64_t meta_offset = 0;

  for (uint32_t i = 0; i < num_vectors; ++i) {
    const ExactType* vec_start = exact_input.data() + i * vector_size;
    auto encoded =
        AlpRdCompression<TypeParam>::CompressVector(vec_start, vector_size, preset);

    AlpRdVectorMetadata meta = encoded.GetVectorMetadata();
    meta.Store({metadata_buffer.data() + meta_offset, AlpRdVectorMetadata::GetStoredSize()});
    meta_offset += AlpRdVectorMetadata::GetStoredSize();
  }

  // Load metadata cache
  AlpRdMetadataCache<TypeParam> cache = AlpRdMetadataCache<TypeParam>::Load(
      num_vectors, vector_size, total_elements,
      {metadata_buffer.data(), metadata_buffer.size()});

  // Verify cache
  EXPECT_EQ(cache.GetNumVectors(), num_vectors);
  for (uint32_t i = 0; i < num_vectors; ++i) {
    EXPECT_EQ(cache.GetVectorNumElements(i), vector_size);
  }
}

// ============================================================================
// Special Values Tests
// ============================================================================

TEST(AlpRdSpecialValuesTest, Infinity) {
  using ExactType = typename AlpTypedConstants<double>::FloatingToExact;

  std::vector<double> input = {
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      1.0,
      2.0,
      3.0,
  };

  std::vector<ExactType> exact_input(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    std::memcpy(&exact_input[i], &input[i], sizeof(double));
  }

  AlpRdEncodingPreset preset =
      AlpRdCompression<double>::CreateEncodingPreset(exact_input);
  auto encoded = AlpRdCompression<double>::CompressVector(exact_input.data(),
                                                          exact_input.size(), preset);

  std::vector<double> output(input.size());
  AlpRdCompression<double>::DecompressVector(encoded, output.data(),
                                             static_cast<uint16_t>(input.size()));

  // Verify bit-exact reproduction
  for (size_t i = 0; i < input.size(); ++i) {
    EXPECT_EQ(std::memcmp(&output[i], &input[i], sizeof(double)), 0)
        << "Bit mismatch at index " << i;
  }
}

TEST(AlpRdSpecialValuesTest, NaN) {
  using ExactType = typename AlpTypedConstants<double>::FloatingToExact;

  std::vector<double> input = {
      std::nan(""),
      1.0,
      2.0,
      3.0,
      std::nan("1"),
  };

  std::vector<ExactType> exact_input(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    std::memcpy(&exact_input[i], &input[i], sizeof(double));
  }

  AlpRdEncodingPreset preset =
      AlpRdCompression<double>::CreateEncodingPreset(exact_input);
  auto encoded = AlpRdCompression<double>::CompressVector(exact_input.data(),
                                                          exact_input.size(), preset);

  std::vector<double> output(input.size());
  AlpRdCompression<double>::DecompressVector(encoded, output.data(),
                                             static_cast<uint16_t>(input.size()));

  // Verify bit-exact reproduction (NaN bit patterns should be preserved)
  for (size_t i = 0; i < input.size(); ++i) {
    EXPECT_EQ(std::memcmp(&output[i], &input[i], sizeof(double)), 0)
        << "Bit mismatch at index " << i;
  }
}

TEST(AlpRdSpecialValuesTest, NegativeZero) {
  using ExactType = typename AlpTypedConstants<double>::FloatingToExact;

  std::vector<double> input = {-0.0, 0.0, 1.0, -0.0, 2.0};

  std::vector<ExactType> exact_input(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    std::memcpy(&exact_input[i], &input[i], sizeof(double));
  }

  AlpRdEncodingPreset preset =
      AlpRdCompression<double>::CreateEncodingPreset(exact_input);
  auto encoded = AlpRdCompression<double>::CompressVector(exact_input.data(),
                                                          exact_input.size(), preset);

  std::vector<double> output(input.size());
  AlpRdCompression<double>::DecompressVector(encoded, output.data(),
                                             static_cast<uint16_t>(input.size()));

  // Verify bit-exact reproduction (-0.0 should remain -0.0)
  for (size_t i = 0; i < input.size(); ++i) {
    EXPECT_EQ(std::memcmp(&output[i], &input[i], sizeof(double)), 0)
        << "Bit mismatch at index " << i;
  }
}

}  // namespace alp
}  // namespace util
}  // namespace arrow


