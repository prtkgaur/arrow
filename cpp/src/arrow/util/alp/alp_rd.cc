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

#include "arrow/util/alp/alp_rd.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "arrow/util/alp/alp_constants.h"
#include "arrow/util/bit_stream_utils_internal.h"
#include "arrow/util/logging.h"
#include "arrow/util/ubsan.h"

namespace arrow {
namespace util {
namespace alp {

// ----------------------------------------------------------------------
// Internal helper struct

namespace {

/// \brief Helper struct for tracking left part frequency during dictionary building
template <typename T>
struct AlpRdLeftPartInfo {
  uint32_t count;
  T hash;
  AlpRdLeftPartInfo(uint32_t count, T hash) : count(count), hash(hash) {}
};

}  // namespace

// ----------------------------------------------------------------------
// AlpRdEncodingSettings implementation

bool AlpRdEncodingSettings::operator==(const AlpRdEncodingSettings& other) const {
  if (left_bit_width != other.left_bit_width ||
      right_bit_width != other.right_bit_width ||
      num_dict_entries != other.num_dict_entries) {
    return false;
  }
  // Compare dictionary entries
  for (uint16_t i = 0; i < num_dict_entries; ++i) {
    if (flat_dictionary[i] != other.flat_dictionary[i]) {
      return false;
    }
  }
  return true;
}

uint64_t AlpRdEncodingSettings::GetStoredSize() const {
  return GetFixedStoredSize() + GetVariableStoredSize();
}

void AlpRdEncodingSettings::Store(arrow::util::span<char> output_buffer) const {
  const uint64_t stored_size = GetStoredSize();
  ARROW_CHECK(output_buffer.size() >= stored_size)
      << "alprd_encoding_settings_store_size_too_small: " << output_buffer.size()
      << " vs " << stored_size;

  uint64_t offset = 0;

  // left_bit_width: 1 byte
  output_buffer[offset++] = static_cast<char>(left_bit_width);

  // right_bit_width: 1 byte
  output_buffer[offset++] = static_cast<char>(right_bit_width);

  // num_dict_entries: 2 bytes
  std::memcpy(output_buffer.data() + offset, &num_dict_entries, sizeof(num_dict_entries));
  offset += sizeof(num_dict_entries);

  // flat_dictionary: num_dict_entries * 2 bytes
  const uint64_t dict_size = num_dict_entries * sizeof(uint16_t);
  std::memcpy(output_buffer.data() + offset, flat_dictionary.data(), dict_size);
  offset += dict_size;

  ARROW_CHECK(offset == stored_size)
      << "alprd_encoding_settings_store_size_mismatch: " << offset << " vs "
      << stored_size;
}

AlpRdEncodingSettings AlpRdEncodingSettings::Load(
    arrow::util::span<const char> input_buffer) {
  ARROW_CHECK(input_buffer.size() >= GetFixedStoredSize())
      << "alprd_encoding_settings_load_size_too_small: " << input_buffer.size();

  AlpRdEncodingSettings settings;
  uint64_t offset = 0;

  // left_bit_width: 1 byte
  settings.left_bit_width = static_cast<uint8_t>(input_buffer[offset++]);

  // right_bit_width: 1 byte
  settings.right_bit_width = static_cast<uint8_t>(input_buffer[offset++]);

  // num_dict_entries: 2 bytes
  std::memcpy(&settings.num_dict_entries, input_buffer.data() + offset,
              sizeof(settings.num_dict_entries));
  offset += sizeof(settings.num_dict_entries);

  // Validate buffer has enough space for dictionary
  const uint64_t dict_size = settings.num_dict_entries * sizeof(uint16_t);
  ARROW_CHECK(input_buffer.size() >= offset + dict_size)
      << "alprd_encoding_settings_load_dict_too_small: " << input_buffer.size()
      << " vs " << (offset + dict_size);

  // Resize and load flat_dictionary
  // +1 for exception marker slot
  settings.flat_dictionary.resize(settings.num_dict_entries + 1);
  std::memcpy(settings.flat_dictionary.data(), input_buffer.data() + offset, dict_size);

  return settings;
}

// ----------------------------------------------------------------------
// AlpRdEncodedVectorInfo implementation

bool AlpRdEncodedVectorInfo::operator==(const AlpRdEncodedVectorInfo& other) const {
  return settings == other.settings &&
         right_bit_packed_size == other.right_bit_packed_size &&
         left_bit_packed_size == other.left_bit_packed_size &&
         num_exceptions == other.num_exceptions;
}

uint64_t AlpRdEncodedVectorInfo::GetStoredSize() const {
  // Note: num_elements is NOT stored; it's derived from header by caller.
  return settings.GetStoredSize() + sizeof(left_bit_packed_size) +
         sizeof(right_bit_packed_size) + sizeof(num_exceptions);
}

uint64_t AlpRdEncodedVectorInfo::GetMaxStoredSize(uint8_t max_dictionary_bit_width) {
  // Note: num_elements is NOT stored; it's derived from header by caller.
  const uint64_t max_dict_size =
      AlpRdConstants::GetMaxDictionarySize(max_dictionary_bit_width) * sizeof(uint16_t);
  return AlpRdEncodingSettings::GetFixedStoredSize() + max_dict_size +
         sizeof(uint64_t) +  // left_bit_packed_size
         sizeof(uint64_t) +  // right_bit_packed_size
         sizeof(uint16_t);   // num_exceptions
}

void AlpRdEncodedVectorInfo::Store(arrow::util::span<char> output_buffer) const {
  uint64_t output_offset = 0;

  const uint64_t overall_size = GetStoredSize();
  ARROW_CHECK(output_buffer.size() >= overall_size)
      << "alprd_encoded_vector_info_store_size_too_small: " << output_buffer.size()
      << " vs " << overall_size;

  settings.Store(output_buffer);
  output_offset += settings.GetStoredSize();

  std::memcpy(output_buffer.data() + output_offset, &left_bit_packed_size,
              sizeof(left_bit_packed_size));
  output_offset += sizeof(left_bit_packed_size);

  std::memcpy(output_buffer.data() + output_offset, &right_bit_packed_size,
              sizeof(right_bit_packed_size));
  output_offset += sizeof(right_bit_packed_size);

  // Note: num_elements is NOT stored; it's derived from header by caller.

  std::memcpy(output_buffer.data() + output_offset, &num_exceptions,
              sizeof(num_exceptions));
  output_offset += sizeof(num_exceptions);
}

AlpRdEncodedVectorInfo AlpRdEncodedVectorInfo::Load(
    arrow::util::span<const char> input_buffer) {
  uint64_t input_offset = 0;

  AlpRdEncodedVectorInfo out;

  out.settings = AlpRdEncodingSettings::Load(input_buffer);
  input_offset += out.settings.GetStoredSize();

  ARROW_CHECK(input_buffer.size() >= out.GetStoredSize())
      << "alprd_encoded_vector_info_total_size: " << input_buffer.size() << " vs "
      << out.GetStoredSize();

  std::memcpy(&out.left_bit_packed_size, input_buffer.data() + input_offset,
              sizeof(left_bit_packed_size));
  input_offset += sizeof(left_bit_packed_size);

  std::memcpy(&out.right_bit_packed_size, input_buffer.data() + input_offset,
              sizeof(right_bit_packed_size));
  input_offset += sizeof(right_bit_packed_size);

  // Note: num_elements is NOT stored; it's derived from header by caller.

  std::memcpy(&out.num_exceptions, input_buffer.data() + input_offset,
              sizeof(num_exceptions));

  return out;
}

// ----------------------------------------------------------------------
// AlpRdVectorMetadata implementation

void AlpRdVectorMetadata::Store(arrow::util::span<char> output_buffer) const {
  ARROW_CHECK(output_buffer.size() >= GetStoredSize())
      << "alprd_vector_metadata_store_size_too_small: " << output_buffer.size()
      << " vs " << GetStoredSize();

  uint64_t offset = 0;
  std::memcpy(output_buffer.data() + offset, &left_bit_width, sizeof(left_bit_width));
  offset += sizeof(left_bit_width);

  std::memcpy(output_buffer.data() + offset, &right_bit_width, sizeof(right_bit_width));
  offset += sizeof(right_bit_width);

  std::memcpy(output_buffer.data() + offset, &num_exceptions, sizeof(num_exceptions));
  offset += sizeof(num_exceptions);

  std::memcpy(output_buffer.data() + offset, &left_bit_packed_size,
              sizeof(left_bit_packed_size));
  offset += sizeof(left_bit_packed_size);

  std::memcpy(output_buffer.data() + offset, &right_bit_packed_size,
              sizeof(right_bit_packed_size));
  offset += sizeof(right_bit_packed_size);

  ARROW_CHECK(offset == GetStoredSize())
      << "alprd_vector_metadata_store_size_mismatch: " << offset << " vs "
      << GetStoredSize();
}

AlpRdVectorMetadata AlpRdVectorMetadata::Load(
    arrow::util::span<const char> input_buffer) {
  ARROW_CHECK(input_buffer.size() >= GetStoredSize())
      << "alprd_vector_metadata_load_size_too_small: " << input_buffer.size()
      << " vs " << GetStoredSize();

  AlpRdVectorMetadata meta;
  uint64_t offset = 0;

  std::memcpy(&meta.left_bit_width, input_buffer.data() + offset,
              sizeof(meta.left_bit_width));
  offset += sizeof(meta.left_bit_width);

  std::memcpy(&meta.right_bit_width, input_buffer.data() + offset,
              sizeof(meta.right_bit_width));
  offset += sizeof(meta.right_bit_width);

  std::memcpy(&meta.num_exceptions, input_buffer.data() + offset,
              sizeof(meta.num_exceptions));
  offset += sizeof(meta.num_exceptions);

  std::memcpy(&meta.left_bit_packed_size, input_buffer.data() + offset,
              sizeof(meta.left_bit_packed_size));
  offset += sizeof(meta.left_bit_packed_size);

  std::memcpy(&meta.right_bit_packed_size, input_buffer.data() + offset,
              sizeof(meta.right_bit_packed_size));
  offset += sizeof(meta.right_bit_packed_size);

  return meta;
}

// ----------------------------------------------------------------------
// AlpRdEncodedVectorData implementation

template <typename T>
AlpRdEncodedVectorData<T> AlpRdEncodedVectorData<T>::LoadView(
    arrow::util::span<const char> data_buffer, const AlpRdVectorMetadata& meta) {
  AlpRdEncodedVectorData<T> data;
  uint64_t offset = 0;

  // Zero-copy views for packed values (bytes have no alignment requirements)
  data.left_parts_encoded = arrow::util::span<const uint8_t>(
      reinterpret_cast<const uint8_t*>(data_buffer.data() + offset),
      meta.left_bit_packed_size);
  offset += meta.left_bit_packed_size;

  data.right_parts_encoded = arrow::util::span<const uint8_t>(
      reinterpret_cast<const uint8_t*>(data_buffer.data() + offset),
      meta.right_bit_packed_size);
  offset += meta.right_bit_packed_size;

  // Copy exception positions into aligned storage to avoid UB from misaligned access.
  const uint16_t num_exceptions = meta.num_exceptions;
  data.exception_positions.UnsafeResize(num_exceptions);
  std::memcpy(data.exception_positions.data(), data_buffer.data() + offset,
              num_exceptions * sizeof(AlpRdConstants::PositionType));
  offset += num_exceptions * sizeof(AlpRdConstants::PositionType);

  // Copy exception values into aligned storage to avoid UB from misaligned access.
  data.exceptions.UnsafeResize(num_exceptions);
  std::memcpy(data.exceptions.data(), data_buffer.data() + offset,
              num_exceptions * sizeof(AlpRdConstants::ExceptionType));

  return data;
}

template <typename T>
uint64_t AlpRdEncodedVectorData<T>::GetSize(const AlpRdVectorMetadata& meta,
                                           [[maybe_unused]] uint16_t num_elements) {
  // Data size = leftParts + rightParts + exception positions + exception values
  return meta.left_bit_packed_size + meta.right_bit_packed_size +
         meta.num_exceptions * sizeof(AlpRdConstants::PositionType) +
         meta.num_exceptions * sizeof(AlpRdConstants::ExceptionType);
}

template struct AlpRdEncodedVectorData<float>;
template struct AlpRdEncodedVectorData<double>;

// ----------------------------------------------------------------------
// AlpRdEncodedVector implementation

template <typename T>
uint64_t AlpRdEncodedVector<T>::GetStoredSize() const {
  // Note: num_elements is stored in the vector, not vector_info, and is not serialized.
  return vector_info.GetStoredSize() + vector_info.left_bit_packed_size +
         vector_info.right_bit_packed_size +
         vector_info.num_exceptions * (sizeof(AlpRdConstants::ExceptionType) +
                                       sizeof(AlpRdConstants::PositionType));
}

template <typename T>
uint64_t AlpRdEncodedVector<T>::GetMaxStoredSize(uint8_t max_dictionary_bit_width) {
  return AlpRdEncodedVectorInfo::GetMaxStoredSize(max_dictionary_bit_width) +
         AlpRdConstants::kAlpVectorSize * sizeof(T) +  // right_parts max
         AlpRdConstants::kAlpVectorSize * sizeof(T) +  // left_parts max
         AlpRdConstants::kAlpVectorSize *
             (sizeof(AlpRdConstants::ExceptionType) +
              sizeof(AlpRdConstants::PositionType));
}

template <typename T>
bool AlpRdEncodedVector<T>::operator==(const AlpRdEncodedVector<T>& other) const {
  const bool vector_info_equal = vector_info == other.vector_info;

  // Compare StaticVector contents manually since StaticVector doesn't have operator==
  const bool left_size_equal = left_parts_encoded.size() == other.left_parts_encoded.size();
  const bool left_bit_packed_data_equal =
      left_size_equal &&
      (left_parts_encoded.empty() ||
       std::memcmp(left_parts_encoded.data(), other.left_parts_encoded.data(),
                   left_parts_encoded.size()) == 0);

  const bool right_size_equal =
      right_parts_encoded.size() == other.right_parts_encoded.size();
  const bool right_bit_packed_data_equal =
      right_size_equal &&
      (right_parts_encoded.empty() ||
       std::memcmp(right_parts_encoded.data(), other.right_parts_encoded.data(),
                   right_parts_encoded.size()) == 0);

  return vector_info_equal && left_bit_packed_data_equal && right_bit_packed_data_equal;
}

template <typename T>
void AlpRdEncodedVector<T>::Store(arrow::util::span<char> output_buffer) const {
  uint64_t output_offset = 0;

  const uint64_t overall_size = GetStoredSize();
  // Stop compression if output is too small.
  if (output_buffer.size() < overall_size) {
    return;
  }

  // Store vector info
  vector_info.Store(output_buffer);
  output_offset += vector_info.GetStoredSize();

  // Then, store left side and right side
  std::memcpy(output_buffer.data() + output_offset, left_parts_encoded.data(),
              vector_info.left_bit_packed_size);
  output_offset += vector_info.left_bit_packed_size;
  std::memcpy(output_buffer.data() + output_offset, right_parts_encoded.data(),
              vector_info.right_bit_packed_size);
  output_offset += vector_info.right_bit_packed_size;

  ARROW_CHECK(vector_info.num_exceptions == exception_positions.size() &&
              vector_info.num_exceptions == exceptions.size())
      << "alp_rd_store_exception_size_mismatch: " << vector_info.num_exceptions << " vs "
      << exception_positions.size() << " vs " << exceptions.size();

  // Store exceptions, consisting of their positions and their values.
  const uint64_t exception_position_size =
      vector_info.num_exceptions * sizeof(AlpRdConstants::PositionType);
  std::memcpy(output_buffer.data() + output_offset, exception_positions.data(),
              exception_position_size);
  output_offset += exception_position_size;

  const uint64_t exception_size =
      vector_info.num_exceptions * sizeof(AlpRdConstants::ExceptionType);
  std::memcpy(output_buffer.data() + output_offset, exceptions.data(), exception_size);
  output_offset += exception_size;

  ARROW_CHECK(output_offset == overall_size)
      << "alp_rd_store_size_mismatch: " << output_offset << " vs " << overall_size;
}

template <typename T>
AlpRdEncodedVector<T> AlpRdEncodedVector<T>::Load(
    arrow::util::span<const char> input_buffer, uint16_t num_elements) {
  uint64_t input_offset = 0;

  AlpRdEncodedVector<T> encoded_vector;
  encoded_vector.vector_info = AlpRdEncodedVectorInfo::Load(input_buffer);
  encoded_vector.num_elements = num_elements;
  input_offset += encoded_vector.vector_info.GetStoredSize();

  const AlpRdEncodedVectorInfo& vector_info = encoded_vector.vector_info;

  const uint64_t overall_size = encoded_vector.GetStoredSize();

  ARROW_CHECK(input_buffer.size() >= overall_size)
      << "alp_rd_load_input_too_small: " << input_buffer.size() << " vs " << overall_size;
  ARROW_CHECK(num_elements <= AlpRdConstants::kAlpVectorSize)
      << "alp_rd_compression_state_element_count_too_large: " << num_elements << " vs "
      << AlpRdConstants::kAlpVectorSize;

  // Load left side and right side
  encoded_vector.left_parts_encoded.UnsafeResize(vector_info.left_bit_packed_size);
  std::memcpy(encoded_vector.left_parts_encoded.data(), input_buffer.data() + input_offset,
              vector_info.left_bit_packed_size);
  input_offset += vector_info.left_bit_packed_size;

  encoded_vector.right_parts_encoded.UnsafeResize(vector_info.right_bit_packed_size);
  std::memcpy(encoded_vector.right_parts_encoded.data(),
              input_buffer.data() + input_offset, vector_info.right_bit_packed_size);
  input_offset += vector_info.right_bit_packed_size;

  // Load exceptions
  const uint64_t exception_position_size =
      vector_info.num_exceptions * sizeof(AlpRdConstants::PositionType);
  encoded_vector.exception_positions.UnsafeResize(vector_info.num_exceptions);
  std::memcpy(encoded_vector.exception_positions.data(),
              input_buffer.data() + input_offset, exception_position_size);
  input_offset += exception_position_size;

  encoded_vector.exceptions.UnsafeResize(vector_info.num_exceptions);
  const uint64_t exception_size =
      vector_info.num_exceptions * sizeof(AlpRdConstants::ExceptionType);
  std::memcpy(encoded_vector.exceptions.data(), input_buffer.data() + input_offset,
              exception_size);
  input_offset += exception_size;

  return encoded_vector;
}

template <typename T>
uint64_t AlpRdEncodedVector<T>::GetDataSize() const {
  return vector_info.left_bit_packed_size + vector_info.right_bit_packed_size +
         vector_info.num_exceptions * (sizeof(AlpRdConstants::ExceptionType) +
                                       sizeof(AlpRdConstants::PositionType));
}

template <typename T>
void AlpRdEncodedVector<T>::StoreDataOnly(arrow::util::span<char> output_buffer) const {
  uint64_t output_offset = 0;

  const uint64_t data_size = GetDataSize();
  ARROW_CHECK(output_buffer.size() >= data_size)
      << "alprd_store_data_only_size_too_small: " << output_buffer.size() << " vs "
      << data_size;

  // Store left parts
  std::memcpy(output_buffer.data() + output_offset, left_parts_encoded.data(),
              vector_info.left_bit_packed_size);
  output_offset += vector_info.left_bit_packed_size;

  // Store right parts
  std::memcpy(output_buffer.data() + output_offset, right_parts_encoded.data(),
              vector_info.right_bit_packed_size);
  output_offset += vector_info.right_bit_packed_size;

  // Store exception positions
  const uint64_t exception_position_size =
      vector_info.num_exceptions * sizeof(AlpRdConstants::PositionType);
  std::memcpy(output_buffer.data() + output_offset, exception_positions.data(),
              exception_position_size);
  output_offset += exception_position_size;

  // Store exception values
  const uint64_t exception_size =
      vector_info.num_exceptions * sizeof(AlpRdConstants::ExceptionType);
  std::memcpy(output_buffer.data() + output_offset, exceptions.data(), exception_size);
  output_offset += exception_size;

  ARROW_CHECK(output_offset == data_size)
      << "alprd_store_data_only_size_mismatch: " << output_offset << " vs " << data_size;
}

template class AlpRdEncodedVector<float>;
template class AlpRdEncodedVector<double>;

// ----------------------------------------------------------------------
// AlpRdMetadataCache implementation

template <typename T>
AlpRdMetadataCache<T> AlpRdMetadataCache<T>::Load(
    uint32_t num_vectors, uint32_t vector_size, uint64_t total_elements,
    arrow::util::span<const char> metadata_buffer) {
  AlpRdMetadataCache<T> cache;
  cache.vector_metadata_.reserve(num_vectors);
  cache.cumulative_data_offsets_.reserve(num_vectors);
  cache.vector_num_elements_.reserve(num_vectors);

  uint64_t meta_offset = 0;
  uint64_t cumulative_data_offset = 0;
  uint64_t remaining_elements = total_elements;

  for (uint32_t i = 0; i < num_vectors; ++i) {
    // Determine number of elements in this vector
    const uint16_t num_elements = static_cast<uint16_t>(
        std::min(static_cast<uint64_t>(vector_size), remaining_elements));
    cache.vector_num_elements_.push_back(num_elements);
    remaining_elements -= num_elements;

    // Load metadata for this vector
    const AlpRdVectorMetadata meta =
        AlpRdVectorMetadata::Load(metadata_buffer.subspan(meta_offset));
    cache.vector_metadata_.push_back(meta);
    meta_offset += AlpRdVectorMetadata::GetStoredSize();

    // Store cumulative offset and compute data size
    cache.cumulative_data_offsets_.push_back(cumulative_data_offset);
    cumulative_data_offset += meta.GetDataSize();
  }

  cache.total_data_size_ = cumulative_data_offset;
  return cache;
}

template class AlpRdMetadataCache<float>;
template class AlpRdMetadataCache<double>;

// ----------------------------------------------------------------------
// AlpRdEncodedVectorView implementation

template <typename T>
AlpRdEncodedVectorView<T> AlpRdEncodedVectorView<T>::LoadView(
    arrow::util::span<const char> input_buffer, uint16_t num_elements) {
  AlpRdEncodedVectorView<T> view;

  // Load the small metadata struct
  view.vector_info = AlpRdEncodedVectorInfo::Load(input_buffer);
  view.num_elements = num_elements;

  uint64_t offset = view.vector_info.GetStoredSize();

  // Zero-copy views for packed values (bytes have no alignment requirements)
  view.left_parts_encoded = arrow::util::span<const uint8_t>(
      reinterpret_cast<const uint8_t*>(input_buffer.data() + offset),
      view.vector_info.left_bit_packed_size);
  offset += view.vector_info.left_bit_packed_size;

  view.right_parts_encoded = arrow::util::span<const uint8_t>(
      reinterpret_cast<const uint8_t*>(input_buffer.data() + offset),
      view.vector_info.right_bit_packed_size);
  offset += view.vector_info.right_bit_packed_size;

  // Copy exception positions into aligned storage to avoid UB from misaligned access.
  // Exceptions are rare (typically < 5%), so this copy is negligible.
  const uint16_t num_exceptions = view.vector_info.num_exceptions;
  view.exception_positions.UnsafeResize(num_exceptions);
  std::memcpy(view.exception_positions.data(), input_buffer.data() + offset,
              num_exceptions * sizeof(AlpRdConstants::PositionType));
  offset += num_exceptions * sizeof(AlpRdConstants::PositionType);

  // Copy exception values into aligned storage to avoid UB from misaligned access.
  view.exceptions.UnsafeResize(num_exceptions);
  std::memcpy(view.exceptions.data(), input_buffer.data() + offset,
              num_exceptions * sizeof(AlpRdConstants::ExceptionType));

  return view;
}

template <typename T>
uint64_t AlpRdEncodedVectorView<T>::GetStoredSize() const {
  return vector_info.GetStoredSize() + vector_info.left_bit_packed_size +
         vector_info.right_bit_packed_size +
         vector_info.num_exceptions * (sizeof(AlpRdConstants::ExceptionType) +
                                       sizeof(AlpRdConstants::PositionType));
}

template <typename T>
AlpRdEncodedVectorView<T> AlpRdEncodedVectorView<T>::LoadViewFromSeparateSections(
    const char* metadata_ptr, const char* data_ptr, uint16_t num_elements,
    const AlpRdEncodingSettings& settings) {
  AlpRdEncodedVectorView<T> view;
  view.num_elements = num_elements;

  // Parse metadata directly (inlined from AlpRdVectorMetadata::Load)
  // This avoids function call overhead and intermediate struct creation
  uint64_t meta_offset = 0;
  uint8_t left_bit_width;
  uint8_t right_bit_width;
  uint16_t num_exceptions;
  uint64_t left_bit_packed_size;
  uint64_t right_bit_packed_size;

  std::memcpy(&left_bit_width, metadata_ptr + meta_offset, sizeof(left_bit_width));
  meta_offset += sizeof(left_bit_width);
  std::memcpy(&right_bit_width, metadata_ptr + meta_offset, sizeof(right_bit_width));
  meta_offset += sizeof(right_bit_width);
  std::memcpy(&num_exceptions, metadata_ptr + meta_offset, sizeof(num_exceptions));
  meta_offset += sizeof(num_exceptions);
  std::memcpy(&left_bit_packed_size, metadata_ptr + meta_offset,
              sizeof(left_bit_packed_size));
  meta_offset += sizeof(left_bit_packed_size);
  std::memcpy(&right_bit_packed_size, metadata_ptr + meta_offset,
              sizeof(right_bit_packed_size));

  // Populate vector_info directly
  // Start with page-level settings (contains dictionary) but override bit widths
  // with per-vector values since they may differ
  view.vector_info.settings = settings;
  view.vector_info.settings.left_bit_width = left_bit_width;
  view.vector_info.settings.right_bit_width = right_bit_width;
  view.vector_info.left_bit_packed_size = left_bit_packed_size;
  view.vector_info.right_bit_packed_size = right_bit_packed_size;
  view.vector_info.num_exceptions = num_exceptions;

  // Setup data views directly
  uint64_t data_offset = 0;

  // Zero-copy views for packed values (bytes have no alignment requirements)
  view.left_parts_encoded = arrow::util::span<const uint8_t>(
      reinterpret_cast<const uint8_t*>(data_ptr + data_offset), left_bit_packed_size);
  data_offset += left_bit_packed_size;

  view.right_parts_encoded = arrow::util::span<const uint8_t>(
      reinterpret_cast<const uint8_t*>(data_ptr + data_offset), right_bit_packed_size);
  data_offset += right_bit_packed_size;

  // Copy exception positions into aligned storage to avoid UB from misaligned access.
  view.exception_positions.UnsafeResize(num_exceptions);
  std::memcpy(view.exception_positions.data(), data_ptr + data_offset,
              num_exceptions * sizeof(AlpRdConstants::PositionType));
  data_offset += num_exceptions * sizeof(AlpRdConstants::PositionType);

  // Copy exception values into aligned storage to avoid UB from misaligned access.
  view.exceptions.UnsafeResize(num_exceptions);
  std::memcpy(view.exceptions.data(), data_ptr + data_offset,
              num_exceptions * sizeof(AlpRdConstants::ExceptionType));

  return view;
}

template struct AlpRdEncodedVectorView<float>;
template struct AlpRdEncodedVectorView<double>;

// ----------------------------------------------------------------------
// AlpRdCompression implementation

template <typename T>
double AlpRdCompression<T>::EstimateCompressionSize(const uint8_t right_bit_width,
                                                    const uint8_t left_bit_width,
                                                    const uint16_t num_exceptions,
                                                    const uint64_t sample_count) {
  const uint64_t exceptions_size_bits =
      (num_exceptions *
       ((sizeof(AlpRdConstants::PositionType) + sizeof(AlpRdConstants::ExceptionType)))) *
      8;
  const double estimated_size_bits =
      (right_bit_width + left_bit_width) +
      (exceptions_size_bits / static_cast<double>(sample_count));
  return estimated_size_bits;
}

template <typename T>
double AlpRdCompression<T>::BuildEncodingSettings(const std::vector<ExactType>& values,
                                                  const uint8_t right_bit_width,
                                                  AlpRdEncodingSettings* settings_out) {
  std::unordered_map<ExactType, int32_t> left_parts_hash;
  std::vector<AlpRdLeftPartInfo<ExactType>> left_parts_sorted_repetitions;

  // Building a hash for all the left parts and how many times they appear
  for (const ExactType& value : values) {
    const ExactType left_tmp = value >> right_bit_width;
    left_parts_hash[left_tmp]++;
  }

  // We build a vector from the hash to be able to sort it by repetition count
  left_parts_sorted_repetitions.reserve(left_parts_hash.size());
  for (auto& hash_pair : left_parts_hash) {
    left_parts_sorted_repetitions.emplace_back(hash_pair.second, hash_pair.first);
  }
  std::sort(left_parts_sorted_repetitions.begin(), left_parts_sorted_repetitions.end(),
            [](const AlpRdLeftPartInfo<ExactType>& a,
               const AlpRdLeftPartInfo<ExactType>& b) { return a.count > b.count; });

  // Get max dictionary size based on default bit width
  const uint64_t max_dictionary_size =
      AlpRdConstants::GetMaxDictionarySize(AlpRdConstants::kDefaultMaxDictionaryBitWidth);

  // Exceptions are left parts which do not fit in the fixed dictionary size
  uint32_t num_exceptions = 0;
  for (uint64_t i = max_dictionary_size; i < left_parts_sorted_repetitions.size(); i++) {
    num_exceptions += left_parts_sorted_repetitions[i].count;
  }

  // The left parts bit width after compression is determined by how many elements
  // are in the dictionary
  const uint64_t actual_dictionary_size =
      std::min(max_dictionary_size, left_parts_sorted_repetitions.size());
  const uint8_t left_bit_width =
      std::max(uint8_t{1}, static_cast<uint8_t>(std::ceil(std::log2(actual_dictionary_size))));

  if (settings_out) {
    AlpRdEncodingSettings& settings = *settings_out;
    settings.num_dict_entries = static_cast<uint16_t>(actual_dictionary_size);
    // Resize to +1 because exceptions get num_dict_entries as marker
    settings.flat_dictionary.resize(actual_dictionary_size + 1);
    for (uint64_t dict_idx = 0; dict_idx < actual_dictionary_size; dict_idx++) {
      // The dict keys are mapped to the left part themselves
      settings.flat_dictionary[dict_idx] =
          static_cast<uint16_t>(left_parts_sorted_repetitions[dict_idx].hash);
    }
    settings.left_bit_width = left_bit_width;
    settings.right_bit_width = right_bit_width;

    ARROW_CHECK(settings.left_bit_width > 0 &&
                settings.left_bit_width <= AlpRdConstants::kDefaultMaxDictionaryBitWidth)
        << "alp_rd_malformed_left_bit_width: "
        << static_cast<int>(settings.left_bit_width);
    ARROW_CHECK(settings.right_bit_width > 0)
        << "alp_rd_malformed_right_bit_width: "
        << static_cast<int>(settings.right_bit_width);
    ARROW_CHECK(settings.num_dict_entries <= max_dictionary_size)
        << "alp_rd_malformed_dictionary: " << settings.num_dict_entries << " vs "
        << max_dictionary_size;
  }

  const double estimated_size = EstimateCompressionSize(
      right_bit_width, left_bit_width, static_cast<uint16_t>(num_exceptions),
      values.size());
  return estimated_size;
}

template <typename T>
AlpRdEncodingPreset AlpRdCompression<T>::CreateEncodingPreset(
    const std::vector<ExactType>& values) {
  uint8_t right_bit_width = 0;
  double best_dict_bits_size = std::numeric_limits<double>::max();

  // Finding the best position to CUT the values
  for (uint64_t i = 1; i <= AlpRdConstants::kCuttingLimit; i++) {
    const uint8_t candidate_right_bit_width = static_cast<uint8_t>(kExactTypeBitSize - i);
    const double estimated_bits_size = BuildEncodingSettings(values, candidate_right_bit_width);
    if (estimated_bits_size < best_dict_bits_size) {
      right_bit_width = candidate_right_bit_width;
      best_dict_bits_size = estimated_bits_size;
    }
  }

  AlpRdEncodingSettings settings(AlpRdConstants::kDefaultMaxDictionaryBitWidth);
  BuildEncodingSettings(values, right_bit_width, &settings);
  settings.best_bits_per_value = best_dict_bits_size;
  return settings;
}

template <typename T>
AlpRdEncodedVector<T> AlpRdCompression<T>::CompressVector(
    const ExactType* input_vector, const uint64_t num_elements,
    const AlpRdEncodingPreset& preset) {
  const AlpRdEncodingSettings& settings = preset;

  // We store the right parts in uint64_t instead of ExactType for processing.
  // This does not matter since they are bitpacked afterwards.
  std::array<uint64_t, AlpRdConstants::kAlpVectorSize> right_parts{};
  std::array<uint16_t, AlpRdConstants::kAlpVectorSize> left_parts{};

  // Cutting the floating point values
  for (uint64_t i = 0; i < num_elements; i++) {
    const ExactType tmp = input_vector[i];
    right_parts[i] = tmp & ((1ULL << settings.right_bit_width) - 1);
    left_parts[i] = static_cast<uint16_t>(tmp >> settings.right_bit_width);
  }

  // Create a map from the flat dictionary to improve performance.
  const uint64_t num_dict_entries = preset.num_dict_entries;
  std::unordered_map<uint16_t, uint16_t> dict;
  for (uint64_t i = 0; i < num_dict_entries; i++) {
    dict[preset.flat_dictionary[i]] = static_cast<uint16_t>(i);
  }

  AlpRdEncodedVector<T> encoded_vector;

  // Dictionary encoding for left parts
  for (uint64_t i = 0; i < num_elements; i++) {
    uint16_t dictionary_index;
    const uint16_t dictionary_key = left_parts[i];

    auto it = dict.find(dictionary_key);
    if (it != dict.end()) {
      dictionary_index = it->second;
    } else {
      // If not found on the dictionary we store the smallest non-key index as
      // exception (the dict size)
      dictionary_index = static_cast<uint16_t>(num_dict_entries);
    }
    left_parts[i] = dictionary_index;

    // Left parts not found in the dictionary are stored as exceptions
    if (dictionary_index >= num_dict_entries) {
      encoded_vector.exceptions.push_back(dictionary_key);
      encoded_vector.exception_positions.push_back(static_cast<uint16_t>(i));

      encoded_vector.vector_info.num_exceptions++;
    }
  }

  // ==========================================================================
  // ALP-RD Bit-Packing (distinct from ALP's FOR+BitPack encoding)
  //
  // ALP-RD uses a different encoding scheme than standard ALP:
  // - ALP uses: Decimal Encoding → FOR (Frame of Reference) → Bit-Packing
  // - ALP-RD uses: Left-Right Split → Dictionary Encoding (left) → Bit-Packing
  //
  // The left parts (dictionary indices) and right parts are bit-packed separately.
  // ==========================================================================
  const uint32_t left_bit_packed_size =
      static_cast<uint32_t>(std::ceil((settings.left_bit_width * num_elements) / 8.0));
  encoded_vector.left_parts_encoded.UnsafeResize(left_bit_packed_size);
  {
    arrow::bit_util::BitWriter writer(encoded_vector.left_parts_encoded.data(),
                                      static_cast<int>(left_bit_packed_size));
    for (uint64_t i = 0; i < num_elements; ++i) {
      writer.PutValue(static_cast<uint64_t>(left_parts[i]), settings.left_bit_width);
    }
    writer.Flush(false);
  }

  const uint32_t right_bit_packed_size =
      static_cast<uint32_t>(std::ceil((settings.right_bit_width * num_elements) / 8.0));
  encoded_vector.right_parts_encoded.UnsafeResize(right_bit_packed_size);
  {
    arrow::bit_util::BitWriter writer(encoded_vector.right_parts_encoded.data(),
                                      static_cast<int>(right_bit_packed_size));
    for (uint64_t i = 0; i < num_elements; ++i) {
      writer.PutValue(right_parts[i], settings.right_bit_width);
    }
    writer.Flush(false);
  }

  // Note: num_elements is stored in the vector, NOT in vector_info.
  // It is NOT serialized; the caller derives it from header during decode.
  encoded_vector.num_elements = static_cast<uint16_t>(num_elements);
  encoded_vector.vector_info.left_bit_packed_size = left_bit_packed_size;
  encoded_vector.vector_info.right_bit_packed_size = right_bit_packed_size;
  encoded_vector.vector_info.settings = settings;
  return encoded_vector;
}

template <typename T>
template <typename TargetType>
void AlpRdCompression<T>::DecompressVector(const AlpRdEncodedVector<T>& input_vector,
                                           TargetType* output_vector,
                                           uint16_t num_elements) {
  static_assert(sizeof(T) <= sizeof(TargetType));
  const AlpRdEncodedVectorInfo& vector_info = input_vector.vector_info;
  const AlpRdEncodingSettings& settings = vector_info.settings;

  std::array<uint16_t, AlpRdConstants::kAlpVectorSize> left_decoded{};
  std::array<uint64_t, AlpRdConstants::kAlpVectorSize> right_decoded{};

  // ==========================================================================
  // ALP-RD Bit-Unpacking (reverse of encoding)
  // Unpack left parts (dictionary indices) and right parts separately.
  // ==========================================================================
  {
    arrow::bit_util::BitReader reader(
        input_vector.left_parts_encoded.data(),
        static_cast<int>(input_vector.left_parts_encoded.size()));
    for (uint16_t i = 0; i < num_elements; ++i) {
      uint64_t value = 0;
      reader.GetValue(settings.left_bit_width, &value);
      left_decoded[i] = static_cast<uint16_t>(value);
    }
  }

  {
    arrow::bit_util::BitReader reader(
        input_vector.right_parts_encoded.data(),
        static_cast<int>(input_vector.right_parts_encoded.size()));
    for (uint16_t i = 0; i < num_elements; ++i) {
      uint64_t value = 0;
      reader.GetValue(settings.right_bit_width, &value);
      right_decoded[i] = value;
    }
  }

  auto* left_parts = left_decoded.data();
  auto* right_parts = right_decoded.data();

  // Decoding
  for (uint64_t i = 0; i < num_elements; i++) {
    const uint16_t left = settings.flat_dictionary[left_parts[i]];
    const ExactType right = static_cast<ExactType>(right_parts[i]);
    const ExactType internal_exact =
        (static_cast<ExactType>(left) << settings.right_bit_width) | right;
    T internal_float;
    std::memcpy(&internal_float, &internal_exact, sizeof(T));
    output_vector[i] = internal_float;
  }

  // Exceptions Patching (exceptions only occur in left parts)
  for (uint64_t i = 0; i < vector_info.num_exceptions; i++) {
    const ExactType right =
        static_cast<ExactType>(right_parts[input_vector.exception_positions[i]]);
    const uint16_t left = input_vector.exceptions[i];
    const ExactType internal_exact =
        (static_cast<ExactType>(left) << settings.right_bit_width) | right;
    T internal_float;
    std::memcpy(&internal_float, &internal_exact, sizeof(T));
    output_vector[input_vector.exception_positions[i]] = internal_float;
  }
}

template <typename T>
template <typename TargetType>
void AlpRdCompression<T>::DecompressVectorView(const AlpRdEncodedVectorView<T>& view,
                                               TargetType* output_vector,
                                               uint16_t num_elements) {
  static_assert(sizeof(T) <= sizeof(TargetType));
  const AlpRdEncodedVectorInfo& vector_info = view.vector_info;
  const AlpRdEncodingSettings& settings = vector_info.settings;

  std::array<uint16_t, AlpRdConstants::kAlpVectorSize> left_decoded{};
  std::array<uint64_t, AlpRdConstants::kAlpVectorSize> right_decoded{};

  // ==========================================================================
  // ALP-RD Bit-Unpacking (zero-copy view version)
  // ==========================================================================
  {
    arrow::bit_util::BitReader reader(view.left_parts_encoded.data(),
                                      static_cast<int>(view.left_parts_encoded.size()));
    for (uint16_t i = 0; i < num_elements; ++i) {
      uint64_t value = 0;
      reader.GetValue(settings.left_bit_width, &value);
      left_decoded[i] = static_cast<uint16_t>(value);
    }
  }

  {
    arrow::bit_util::BitReader reader(view.right_parts_encoded.data(),
                                      static_cast<int>(view.right_parts_encoded.size()));
    for (uint16_t i = 0; i < num_elements; ++i) {
      uint64_t value = 0;
      reader.GetValue(settings.right_bit_width, &value);
      right_decoded[i] = value;
    }
  }

  auto* left_parts = left_decoded.data();
  auto* right_parts = right_decoded.data();

  // Dictionary decoding + reconstruction
  for (uint64_t i = 0; i < num_elements; i++) {
    const uint16_t left = settings.flat_dictionary[left_parts[i]];
    const ExactType right = static_cast<ExactType>(right_parts[i]);
    const ExactType internal_exact =
        (static_cast<ExactType>(left) << settings.right_bit_width) | right;
    T internal_float;
    std::memcpy(&internal_float, &internal_exact, sizeof(T));
    output_vector[i] = internal_float;
  }

  // Exceptions Patching (exceptions only occur in left parts) - uses aligned
  // exception data
  for (uint64_t i = 0; i < vector_info.num_exceptions; i++) {
    const ExactType right =
        static_cast<ExactType>(right_parts[view.exception_positions[i]]);
    const uint16_t left = view.exceptions[i];
    const ExactType internal_exact =
        (static_cast<ExactType>(left) << settings.right_bit_width) | right;
    T internal_float;
    std::memcpy(&internal_float, &internal_exact, sizeof(T));
    output_vector[view.exception_positions[i]] = internal_float;
  }
}

// ----------------------------------------------------------------------
// Template instantiations

template void AlpRdCompression<float>::DecompressVector<float>(
    const AlpRdEncodedVector<float>& packed_vector, float* output, uint16_t num_elements);
template void AlpRdCompression<float>::DecompressVector<double>(
    const AlpRdEncodedVector<float>& packed_vector, double* output,
    uint16_t num_elements);
template void AlpRdCompression<double>::DecompressVector<double>(
    const AlpRdEncodedVector<double>& packed_vector, double* output,
    uint16_t num_elements);

template void AlpRdCompression<float>::DecompressVectorView<float>(
    const AlpRdEncodedVectorView<float>& view, float* output, uint16_t num_elements);
template void AlpRdCompression<float>::DecompressVectorView<double>(
    const AlpRdEncodedVectorView<float>& view, double* output, uint16_t num_elements);
template void AlpRdCompression<double>::DecompressVectorView<double>(
    const AlpRdEncodedVectorView<double>& view, double* output, uint16_t num_elements);

template class AlpRdCompression<float>;
template class AlpRdCompression<double>;

}  // namespace alp
}  // namespace util
}  // namespace arrow

