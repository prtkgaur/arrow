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

// ALP-RD (Real Doubles) compression implementation
//
// IMPORTANT: For abstract interfaces or examples how to use ALP-RD, consult
// alp_wrapper.h.
// This is our implementation of the adaptive lossless floating-point compression
// for real doubles (ALP-RD) (https://dl.acm.org/doi/10.1145/3626717). Despite its
// name, it is able to compress both floats and doubles. It is used as a fallback
// mechanism for ALP if the data is not effectively compressible by decimal
// compression. It works by interpreting the float as an integer and splitting it
// into a left and a right part, dictionary encoding the left part and bit-packing
// the left and right part. Left parts that were not compressible well using
// dictionary encoding are stored uncompressed separately as exceptions and are
// patched into the data again at decompression.

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "arrow/util/alp/alp_constants.h"
#include "arrow/util/logging.h"
#include "arrow/util/small_vector.h"
#include "arrow/util/span.h"

namespace arrow {
namespace util {
namespace alp {

// =============================================================================
// ALP-RD (Real Doubles) Encoding Overview
//
// ALP-RD uses a different encoding method than standard ALP:
//
// Standard ALP:
//   Decimal Encoding → FOR (Frame of Reference) → Bit-Packing
//   Controlled by: AlpIntegerEncoding (e.g., kForBitPack)
//
// ALP-RD:
//   Left-Right Split → Dictionary Encoding (left) → Bit-Packing (both)
//   NOT controlled by AlpIntegerEncoding - uses its own fixed scheme
//
// Key differences:
//   - ALP encodes floats as integers via decimal encoding, then uses FOR
//   - ALP-RD splits the raw bit representation into left (high) and right (low)
//   - ALP-RD uses a dictionary for left parts to achieve compression
//   - Both use bit-packing for the final encoding, but in different ways
//
// The AlpMode field distinguishes between kAlp and kAlpRd in the header.
// =============================================================================

// Forward declarations
template <typename T>
class AlpRdCompression;

template <typename T>
class AlpSampler;

// ----------------------------------------------------------------------
// AlpRdEncodingSettings

/// \brief Encoding settings for ALP-RD compression (dictionary configuration)
///
/// In contrast to ALP, the AlpRdEncodingSettings are not chosen dynamically
/// during compression. They are determined once during the sampling phase
/// and applied to all vectors in the page.
///
/// The dictionary stores the most common left parts (high bits) of the
/// floating-point values. Values whose left parts are in the dictionary
/// can be encoded efficiently; others are stored as exceptions.
///
/// Serialization format:
///   +------------------------------------------+
///   |  AlpRdEncodingSettings Layout            |
///   +------------------------------------------+
///   |  Offset |  Field              |  Size    |
///   +---------+---------------------+----------+
///   |    0    |  left_bit_width     |  1 byte  |
///   |    1    |  right_bit_width    |  1 byte  |
///   |    2    |  num_dict_entries   |  2 bytes |
///   |    4    |  flat_dictionary    |  variable|
///   +------------------------------------------+
struct AlpRdEncodingSettings {
  /// Number of bits used for dictionary indices (left parts after encoding)
  uint8_t left_bit_width = 0;
  /// Number of bits used for right parts
  uint8_t right_bit_width = 0;
  /// Number of entries in the dictionary
  uint16_t num_dict_entries = 0;
  /// The dictionary mapping indices to original left part values
  std::vector<uint16_t> flat_dictionary;
  /// Estimated bits per value for this configuration (not serialized)
  double best_bits_per_value = 0;

  /// \brief Default constructor (for deserialization)
  AlpRdEncodingSettings() = default;

  /// \brief Construct with a specific max dictionary bit width
  ///
  /// \param[in] max_dictionary_bit_width maximum bits for dictionary indices
  explicit AlpRdEncodingSettings(uint8_t max_dictionary_bit_width) {
    const uint64_t max_dictionary_size =
        AlpRdConstants::GetMaxDictionarySize(max_dictionary_bit_width);
    // Resize to +1 because exceptions get num_dict_entries+1 as marker
    flat_dictionary.resize(max_dictionary_size + 1);
  }

  /// \brief Get the maximum dictionary size for this configuration
  ///
  /// \return the maximum number of dictionary entries
  uint64_t GetMaxDictionarySize() const {
    return flat_dictionary.empty() ? 0 : flat_dictionary.size() - 1;
  }

  bool operator==(const AlpRdEncodingSettings& other) const;

  /// \brief Get the total serialized size of these settings
  ///
  /// \return size in bytes
  uint64_t GetStoredSize() const;

  /// \brief Serialize settings into an output buffer
  ///
  /// \param[out] output_buffer buffer to write into (must be >= GetStoredSize())
  void Store(arrow::util::span<char> output_buffer) const;

  /// \brief Get the fixed portion size (excludes variable dictionary)
  ///
  /// \return fixed size in bytes (4 bytes)
  static uint64_t GetFixedStoredSize() { return 4; }

  /// \brief Get the variable portion size (dictionary only)
  ///
  /// \return size of dictionary in bytes
  uint64_t GetVariableStoredSize() const {
    return num_dict_entries * sizeof(uint16_t);
  }

  /// \brief Deserialize settings from an input buffer
  ///
  /// \param[in] input_buffer buffer to read from
  /// \return the deserialized settings
  static AlpRdEncodingSettings Load(arrow::util::span<const char> input_buffer);
};

/// Alias for preset (settings are used as the preset)
using AlpRdEncodingPreset = AlpRdEncodingSettings;

// ----------------------------------------------------------------------
// AlpRdEncodedVectorInfo

/// \brief Metadata for a compressed ALP-RD vector (serialized per-vector)
///
/// This struct contains the minimal information needed to decompress an ALP-RD
/// vector. It includes the encoding settings (dictionary) and packed data sizes.
///
/// Note: num_elements is NOT stored in per-vector info because it is redundant:
///   - For all vectors except the last: num_elements = 1 << log_vector_size (1024)
///   - For the last vector: num_elements = total_elements % vector_size
///     (or vector_size if evenly divisible)
///   The caller must track remaining elements and pass num_elements to functions.
struct AlpRdEncodedVectorInfo {
  AlpRdEncodingSettings settings;
  uint64_t right_bit_packed_size = 0;
  uint64_t left_bit_packed_size = 0;
  uint16_t num_exceptions = 0;

  bool operator==(const AlpRdEncodedVectorInfo& other) const;

  /// \brief Get the total serialized size
  ///
  /// Note: Requires settings.left_bit_width to be set for accurate size.
  ///
  /// \return size in bytes
  uint64_t GetStoredSize() const;

  /// \brief Serialize the info into the output buffer
  ///
  /// \param[out] output_buffer buffer to write into
  void Store(arrow::util::span<char> output_buffer) const;

  /// \brief Get the maximum possible serialized size
  ///
  /// \param[in] max_dictionary_bit_width maximum bits for dictionary indices
  /// \return maximum size in bytes
  static uint64_t GetMaxStoredSize(uint8_t max_dictionary_bit_width);

  /// \brief Deserialize info from the input buffer
  ///
  /// \param[in] input_buffer buffer to read from
  /// \return the deserialized info
  static AlpRdEncodedVectorInfo Load(arrow::util::span<const char> input_buffer);
};

// ----------------------------------------------------------------------
// AlpRdVectorMetadata

/// \brief Fixed-size per-vector metadata for ALP-RD (metadata-first layout)
///
/// Unlike AlpRdEncodedVectorInfo which contains the full dictionary (variable),
/// this struct contains only the fixed-size fields needed per vector.
/// The dictionary is stored once in the header (shared across all vectors).
///
/// Serialization format (20 bytes fixed):
///   +------------------------------------------+
///   |  AlpRdVectorMetadata Layout              |
///   +------------------------------------------+
///   |  Offset |  Field              |  Size    |
///   +---------+---------------------+----------+
///   |    0    |  left_bit_width     |  1 byte  |
///   |    1    |  right_bit_width    |  1 byte  |
///   |    2    |  num_exceptions     |  2 bytes |
///   |    4    |  left_bit_packed    |  8 bytes |
///   |   12    |  right_bit_packed   |  8 bytes |
///   +------------------------------------------+
///
/// Note: num_elements is NOT stored (derived from header).
///
/// Usage:
///   - During encoding: extracted from AlpRdEncodedVector and stored contiguously
///   - During decoding: loaded via AlpRdMetadataCache, combined with dictionary
struct AlpRdVectorMetadata {
  uint8_t left_bit_width = 0;
  uint8_t right_bit_width = 0;
  uint16_t num_exceptions = 0;
  uint64_t left_bit_packed_size = 0;
  uint64_t right_bit_packed_size = 0;

  bool operator==(const AlpRdVectorMetadata& other) const {
    return left_bit_width == other.left_bit_width &&
           right_bit_width == other.right_bit_width &&
           num_exceptions == other.num_exceptions &&
           left_bit_packed_size == other.left_bit_packed_size &&
           right_bit_packed_size == other.right_bit_packed_size;
  }

  /// \brief Get the fixed serialized size
  ///
  /// \return 20 bytes
  static constexpr uint64_t GetStoredSize() { return 20; }

  /// \brief Serialize into output buffer
  ///
  /// \param[out] output_buffer buffer to write into
  void Store(arrow::util::span<char> output_buffer) const;

  /// \brief Deserialize from input buffer
  ///
  /// \param[in] input_buffer buffer to read from
  /// \return the deserialized metadata
  static AlpRdVectorMetadata Load(arrow::util::span<const char> input_buffer);

  /// \brief Create from AlpRdEncodedVectorInfo (extracts fixed parts)
  ///
  /// \param[in] info the full vector info
  /// \return the fixed-size metadata
  static AlpRdVectorMetadata FromVectorInfo(const AlpRdEncodedVectorInfo& info) {
    AlpRdVectorMetadata meta;
    meta.left_bit_width = info.settings.left_bit_width;
    meta.right_bit_width = info.settings.right_bit_width;
    meta.num_exceptions = info.num_exceptions;
    meta.left_bit_packed_size = info.left_bit_packed_size;
    meta.right_bit_packed_size = info.right_bit_packed_size;
    return meta;
  }

  /// \brief Compute size of data portion from this metadata
  ///
  /// \return size in bytes of [left_packed][right_packed][positions][exceptions]
  uint64_t GetDataSize() const {
    return left_bit_packed_size + right_bit_packed_size +
           num_exceptions * (sizeof(AlpRdConstants::PositionType) +
                             sizeof(AlpRdConstants::ExceptionType));
  }
};

// ----------------------------------------------------------------------
// AlpRdEncodedVector

/// \class AlpRdEncodedVector
/// \brief A compressed ALP-RD vector that owns its data
///
/// This class stores the compressed representation of a single vector (up to
/// 1024 elements). ALP-RD splits each value into left (high) and right (low)
/// parts.
///
/// Used during encoding and when the compressed data needs to be owned.
/// For zero-copy decompression, use AlpRdEncodedVectorView instead.
///
/// Serialized layout (via Store()/Load()):
///   +------------------------------------------+
///   |  AlpRdEncodedVectorInfo (variable size)  |
///   +------------------------------------------+
///   |  left_parts_encoded[left_bit_packed]     |
///   +------------------------------------------+
///   |  right_parts_encoded[right_bit_packed]   |
///   +------------------------------------------+
///   |  exception_positions[num_exceptions]     |
///   +------------------------------------------+
///   |  exceptions[num_exceptions]              |
///   +------------------------------------------+
///
/// Note: num_elements is NOT serialized. During deserialization, num_elements
/// is passed as a parameter because it can be derived from the header.
///
/// \tparam T the floating-point type (float or double)
template <typename T>
class AlpRdEncodedVector {
 public:
  using ExactType = typename AlpTypedConstants<T>::FloatingToExact;

  bool operator==(const AlpRdEncodedVector<T>& other) const;

  /// \brief Get the size in bytes if stored in a compact format
  ///
  /// \return the stored size in bytes
  uint64_t GetStoredSize() const;

  /// \brief Store the compressed vector in a compact format
  ///
  /// \param[out] output_buffer buffer to write into
  void Store(arrow::util::span<char> output_buffer) const;

  /// \brief Get the maximum stored size of an AlpRdEncodedVector
  ///
  /// \param[in] max_dictionary_bit_width maximum bits for dictionary indices
  /// \return maximum size in bytes
  static uint64_t GetMaxStoredSize(uint8_t max_dictionary_bit_width);

  /// \brief Load a compressed vector from a compact format
  ///
  /// \param[in] input_buffer buffer to read from
  /// \param[in] num_elements number of elements (derived from header by caller)
  /// \return the loaded AlpRdEncodedVector
  static AlpRdEncodedVector Load(arrow::util::span<const char> input_buffer,
                                 uint16_t num_elements);

  /// \brief Get size of just the data portion (excluding metadata)
  ///
  /// Used for metadata-first layout where metadata is stored separately.
  ///
  /// \return size in bytes
  uint64_t GetDataSize() const;

  /// \brief Store only the data portion (no metadata)
  ///
  /// Writes: [left_parts_encoded][right_parts_encoded][positions][exceptions]
  /// Used for metadata-first layout where metadata is written separately.
  ///
  /// \param[out] output_buffer buffer to write into
  void StoreDataOnly(arrow::util::span<char> output_buffer) const;

  /// \brief Get the fixed-size metadata for this vector
  ///
  /// Used for metadata-first layout to extract per-vector metadata.
  ///
  /// \return the vector metadata
  AlpRdVectorMetadata GetVectorMetadata() const {
    return AlpRdVectorMetadata::FromVectorInfo(vector_info);
  }

  /// Metadata of the compressed vector
  AlpRdEncodedVectorInfo vector_info;
  /// Number of elements in this vector (not serialized; derived from header)
  uint16_t num_elements = 0;
  /// Bitpacked right integer parts of the compressed vector
  arrow::internal::StaticVector<uint8_t, AlpRdConstants::kAlpVectorSize * sizeof(T)>
      right_parts_encoded;
  /// Bitpacked left dictionary indices of the compressed vector
  arrow::internal::StaticVector<uint8_t, AlpRdConstants::kAlpVectorSize * sizeof(T)>
      left_parts_encoded;
  /// Left parts of the vector that were not encodable by dictionary
  arrow::internal::StaticVector<uint16_t, AlpRdConstants::kAlpVectorSize> exceptions;
  /// Positions of the left-part exceptions above in the vector
  arrow::internal::StaticVector<uint16_t, AlpRdConstants::kAlpVectorSize>
      exception_positions;
};

// ----------------------------------------------------------------------
// AlpRdEncodedVectorView

/// \class AlpRdEncodedVectorView
/// \brief A view into compressed ALP-RD data optimized for decompression
///
/// Unlike AlpRdEncodedVector which copies all data into internal buffers,
/// AlpRdEncodedVectorView uses zero-copy for the large packed arrays
/// while copying the small exception arrays into aligned storage.
///
/// The packed values are accessed via spans (zero-copy) since they are
/// byte arrays with no alignment requirements. Exception positions and
/// values are copied into aligned StaticVectors because:
///   1. The serialized data may not be properly aligned for uint16_t access
///   2. Exceptions are rare (typically < 5%), so copying is negligible
///   3. This avoids undefined behavior from misaligned memory access
///
/// Use LoadView() to create a view, then pass to DecompressVectorView().
/// The underlying buffer must remain valid for the lifetime of the view.
///
/// \tparam T the floating-point type (float or double)
template <typename T>
struct AlpRdEncodedVectorView {
  /// Metadata of the compressed vector (copied, small struct)
  AlpRdEncodedVectorInfo vector_info;
  /// Number of elements in this vector
  uint16_t num_elements = 0;
  /// View into bit-packed right parts (zero-copy)
  arrow::util::span<const uint8_t> right_parts_encoded;
  /// View into bit-packed left parts (zero-copy)
  arrow::util::span<const uint8_t> left_parts_encoded;
  /// Exception values (copied into aligned storage)
  arrow::internal::StaticVector<uint16_t, AlpRdConstants::kAlpVectorSize> exceptions;
  /// Exception positions (copied into aligned storage)
  arrow::internal::StaticVector<uint16_t, AlpRdConstants::kAlpVectorSize>
      exception_positions;

  /// \brief Create a zero-copy view into compressed ALP-RD data
  ///
  /// \param[in] input_buffer buffer containing the compressed data
  /// \param[in] num_elements number of elements (derived from header by caller)
  /// \return a view pointing into the input buffer
  static AlpRdEncodedVectorView LoadView(arrow::util::span<const char> input_buffer,
                                         uint16_t num_elements);

  /// \brief Create a zero-copy view when metadata and data are separate
  ///
  /// This is optimized for the metadata-first layout. Combines metadata parsing,
  /// data view setup, and view construction into a single call.
  ///
  /// \param[in] metadata_ptr pointer to the metadata for this vector
  /// \param[in] data_ptr pointer to the data for this vector
  /// \param[in] num_elements number of elements in this vector
  /// \param[in] settings the ALP-RD encoding settings (dictionary), shared
  /// \return a view pointing into the provided buffers
  static AlpRdEncodedVectorView LoadViewFromSeparateSections(
      const char* metadata_ptr, const char* data_ptr, uint16_t num_elements,
      const AlpRdEncodingSettings& settings);

  /// \brief Get the total size of the compressed data this view represents
  ///
  /// \return size in bytes
  uint64_t GetStoredSize() const;
};

// ----------------------------------------------------------------------
// AlpRdEncodedVectorData

/// \class AlpRdEncodedVectorData
/// \brief A view into the data portion of a compressed ALP-RD vector
///
/// With the metadata-first layout, vector metadata and data are stored
/// separately. This struct provides a zero-copy view into just the data:
///   - left_parts_encoded: bit-packed dictionary indices
///   - right_parts_encoded: bit-packed right parts
///   - exception_positions: positions of exception values
///   - exceptions: the exception values themselves
///
/// Used during decoding when metadata has been loaded from AlpRdMetadataCache.
///
/// Data layout in serialized form:
///   +------------------------------------------+
///   |  left_parts_encoded[left_bit_packed]     |
///   +------------------------------------------+
///   |  right_parts_encoded[right_bit_packed]   |
///   +------------------------------------------+
///   |  exception_positions[num_exceptions × 2] |
///   +------------------------------------------+
///   |  exceptions[num_exceptions × 2]          |
///   +------------------------------------------+
///
/// \tparam T the floating-point type (float or double)
template <typename T>
struct AlpRdEncodedVectorData {
  /// View into bit-packed left parts (zero-copy)
  arrow::util::span<const uint8_t> left_parts_encoded;
  /// View into bit-packed right parts (zero-copy)
  arrow::util::span<const uint8_t> right_parts_encoded;
  /// Exception positions (copied into aligned storage)
  arrow::internal::StaticVector<uint16_t, AlpRdConstants::kAlpVectorSize>
      exception_positions;
  /// Exception values (copied into aligned storage)
  arrow::internal::StaticVector<uint16_t, AlpRdConstants::kAlpVectorSize> exceptions;

  /// \brief Create a view into the data portion of compressed ALP-RD data
  ///
  /// \param[in] data_buffer buffer containing just the data (no metadata)
  /// \param[in] meta metadata for this vector (loaded separately)
  /// \return a view into the data buffer
  static AlpRdEncodedVectorData LoadView(arrow::util::span<const char> data_buffer,
                                         const AlpRdVectorMetadata& meta);

  /// \brief Compute the serialized size of the data portion for a vector
  ///
  /// \param[in] meta metadata for the vector
  /// \param[in] num_elements number of elements (unused, for API consistency)
  /// \return size in bytes
  static uint64_t GetSize(const AlpRdVectorMetadata& meta, uint16_t num_elements);
};

// ----------------------------------------------------------------------
// AlpRdMetadataCache

/// \class AlpRdMetadataCache
/// \brief Cache for vector metadata to enable O(1) random access to any vector
///
/// With the metadata-first layout, all vector metadata is stored contiguously
/// after the header. This class loads all metadata into memory and precomputes
/// cumulative data offsets, enabling O(1) access to any vector's data.
///
/// The dictionary is stored once in the header and shared across all vectors,
/// so this cache stores only the fixed-size per-vector metadata.
///
/// Usage:
/// \code
///   // Load metadata from compressed buffer
///   AlpRdMetadataCache<T> cache = AlpRdMetadataCache<T>::Load(
///       num_vectors, vector_size, total_elements, metadata_buffer);
///
///   // Access metadata for any vector in O(1)
///   const auto& meta = cache.GetVectorMetadata(vector_idx);
///
///   // Get offset to any vector's data in O(1)
///   uint64_t data_offset = cache.GetVectorDataOffset(vector_idx);
/// \endcode
///
/// \tparam T the floating-point type (float or double)
template <typename T>
class AlpRdMetadataCache {
 public:
  /// \brief Load all metadata from buffer into cache and precompute offsets
  ///
  /// \param[in] num_vectors number of vectors in the block
  /// \param[in] vector_size size of each full vector (typically 1024)
  /// \param[in] total_elements total number of elements across all vectors
  /// \param[in] metadata_buffer buffer containing all vector metadata contiguously
  /// \return a metadata cache with all vector info and precomputed offsets
  static AlpRdMetadataCache Load(uint32_t num_vectors, uint32_t vector_size,
                                 uint64_t total_elements,
                                 arrow::util::span<const char> metadata_buffer);

  /// \brief Get metadata for vector at given index
  ///
  /// \param[in] vector_idx index of the vector (0 to num_vectors-1)
  /// \return reference to the vector's metadata
  const AlpRdVectorMetadata& GetVectorMetadata(uint32_t vector_idx) const {
    ARROW_CHECK(vector_idx < vector_metadata_.size())
        << "vector_index_out_of_range: " << vector_idx;
    return vector_metadata_[vector_idx];
  }

  /// \brief Get bit widths for vector at given index
  ///
  /// \param[in] vector_idx index of the vector (0 to num_vectors-1)
  /// \return pair of (left_bit_width, right_bit_width)
  std::pair<uint8_t, uint8_t> GetBitWidths(uint32_t vector_idx) const {
    ARROW_CHECK(vector_idx < vector_metadata_.size())
        << "vector_index_out_of_range: " << vector_idx;
    const auto& meta = vector_metadata_[vector_idx];
    return {meta.left_bit_width, meta.right_bit_width};
  }

  /// \brief Get offset to vector's data from start of data section
  ///
  /// \param[in] vector_idx index of the vector (0 to num_vectors-1)
  /// \return byte offset from start of data section to this vector's data
  uint64_t GetVectorDataOffset(uint32_t vector_idx) const {
    ARROW_CHECK(vector_idx < cumulative_data_offsets_.size())
        << "vector_index_out_of_range: " << vector_idx;
    return cumulative_data_offsets_[vector_idx];
  }

  /// \brief Get number of elements in vector at given index
  ///
  /// \param[in] vector_idx index of the vector (0 to num_vectors-1)
  /// \return number of elements in this vector
  uint16_t GetVectorNumElements(uint32_t vector_idx) const {
    ARROW_CHECK(vector_idx < vector_num_elements_.size())
        << "vector_index_out_of_range: " << vector_idx;
    return vector_num_elements_[vector_idx];
  }

  /// \brief Get number of exceptions in vector at given index
  ///
  /// \param[in] vector_idx index of the vector (0 to num_vectors-1)
  /// \return number of exceptions in this vector
  uint16_t GetNumExceptions(uint32_t vector_idx) const {
    ARROW_CHECK(vector_idx < vector_metadata_.size())
        << "vector_index_out_of_range: " << vector_idx;
    return vector_metadata_[vector_idx].num_exceptions;
  }

  /// \brief Get number of vectors in the cache
  ///
  /// \return number of vectors
  uint32_t GetNumVectors() const { return static_cast<uint32_t>(vector_metadata_.size()); }

  /// \brief Get total size of the data section in bytes
  ///
  /// \return total data size
  uint64_t GetTotalDataSize() const { return total_data_size_; }

  /// \brief Get total size of the metadata section in bytes
  ///
  /// \return total metadata size (num_vectors * AlpRdVectorMetadata::GetStoredSize())
  uint64_t GetMetadataSectionSize() const {
    return vector_metadata_.size() * AlpRdVectorMetadata::GetStoredSize();
  }

 private:
  std::vector<AlpRdVectorMetadata> vector_metadata_;
  std::vector<uint64_t> cumulative_data_offsets_;  // Offset from data section start
  std::vector<uint16_t> vector_num_elements_;      // Number of elements in each vector
  uint64_t total_data_size_ = 0;                   // Total size of data section
};

// ----------------------------------------------------------------------
// AlpRdCompression

/// \class AlpRdCompression
/// \brief ALP-RD compression and decompression facilities
///
/// AlpRdCompression contains all facilities to compress and decompress floats
/// and doubles with ALP-RD in a vectorized fashion. Use CreateEncodingPreset()
/// first on a sample of the input data, then compress it vector-wise via
/// CompressVector(). To serialize the data, use the facilities provided by
/// AlpRdEncodedVector.
///
/// \tparam T the type of data to be compressed (float or double)
template <typename T>
class AlpRdCompression : private AlpRdConstants {
 public:
  using ExactType = typename AlpTypedConstants<T>::FloatingToExact;

  /// \brief Compress a vector of floating points via ALP-RD
  ///
  /// \param[in] input_vector vector of floats as their exact integer representation
  /// \param[in] num_elements number of values to be compressed
  /// \param[in] preset the encoding preset to be used
  /// \return the compressed ALP-RD vector
  static AlpRdEncodedVector<T> CompressVector(const ExactType* input_vector,
                                              uint64_t num_elements,
                                              const AlpRdEncodingPreset& preset);

  /// \brief Decompress a compressed vector with ALP-RD
  ///
  /// \param[in] input_vector the encoded vector to decompress
  /// \param[out] output_vector the output vector
  /// \param[in] num_elements number of elements (derived from header by caller)
  /// \tparam TargetType the type used for output (may not be narrowing from T)
  template <typename TargetType>
  static void DecompressVector(const AlpRdEncodedVector<T>& input_vector,
                               TargetType* output_vector, uint16_t num_elements);

  /// \brief Decompress using a zero-copy view (faster, no memory allocation)
  ///
  /// \param[in] view the zero-copy view into the compressed data
  /// \param[out] output_vector the output vector
  /// \param[in] num_elements number of elements in this vector
  /// \tparam TargetType the type used for output (may not be narrowing from T)
  template <typename TargetType>
  static void DecompressVectorView(const AlpRdEncodedVectorView<T>& view,
                                   TargetType* output_vector, uint16_t num_elements);

 protected:
  /// \brief Create an encoding preset from samples
  ///
  /// Finds the best right_bit_width and builds the dictionary for left parts.
  ///
  /// \param[in] values the sample values (as exact integer representations)
  /// \return the encoding preset
  static AlpRdEncodingPreset CreateEncodingPreset(const std::vector<ExactType>& values);
  friend AlpSampler<T>;

 private:
  static constexpr uint8_t kExactTypeBitSize = sizeof(ExactType) * 8;

  /// \brief Estimate the compressed size for a set of samples
  ///
  /// \param[in] right_bit_width number of bits for the right part
  /// \param[in] left_bit_width number of bits for dictionary indices
  /// \param[in] num_exceptions number of exceptions encountered
  /// \param[in] sample_count number of samples
  /// \return estimated bits per value
  static double EstimateCompressionSize(uint8_t right_bit_width, uint8_t left_bit_width,
                                        uint16_t num_exceptions, uint64_t sample_count);

  /// \brief Build encoding settings for a given right_bit_width
  ///
  /// \param[in] values the sample values
  /// \param[in] right_bit_width number of bits for the right part
  /// \param[out] settings_out if != nullptr, stores the built settings
  /// \return estimated bits per value for this configuration
  static double BuildEncodingSettings(const std::vector<ExactType>& values,
                                      uint8_t right_bit_width,
                                      AlpRdEncodingSettings* settings_out = nullptr);
};

}  // namespace alp
}  // namespace util
}  // namespace arrow

