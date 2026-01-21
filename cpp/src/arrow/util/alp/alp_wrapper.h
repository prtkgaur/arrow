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

// High-level wrapper interface for ALP compression

#pragma once

#include <cstddef>
#include <optional>

#include "arrow/util/alp/alp.h"
#include "arrow/util/alp/alp_rd.h"
#include "arrow/util/alp/alp_sampler.h"

namespace arrow {
namespace util {
namespace alp {

// ----------------------------------------------------------------------
// AlpWrapper

/// \class AlpWrapper
/// \brief High-level interface for ALP compression
///
/// AlpWrapper is an interface for Adaptive Lossless floating-Point Compression
/// (ALP) (https://dl.acm.org/doi/10.1145/3626717). For encoding, it samples
/// the data and applies either decimal compression (ALP) or real-doubles
/// compression (ALP-RD) based on estimated compression ratios.
///
/// ALP supports two compression modes:
///   - kAlp: Decimal compression for pseudo-decimal floating-point values
///   - kAlpRd: Real doubles compression via bit-split + dictionary encoding
///
/// The appropriate mode is automatically selected during encoding unless
/// explicitly overridden via the enforce_mode parameter.
///
/// \tparam T the floating point type (float or double)
template <typename T>
class AlpWrapper {
 public:
  /// \brief Create a sampling preset from input data
  ///
  /// Samples the input data and generates encoding presets for both ALP and
  /// ALP-RD compression, along with a recommendation for which mode to use.
  /// This can be used to pre-compute the preset once and reuse it for multiple
  /// encode calls via EncodeWithPreset().
  ///
  /// \param[in] decomp pointer to the input data to sample
  /// \param[in] decomp_size size of decomp in bytes.
  ///            This needs to be a multiple of sizeof(T).
  /// \return AlpSamplerResult containing presets and recommended mode
  static AlpSamplerResult CreateSamplingPreset(const T* decomp, size_t decomp_size);

  /// \brief Encode floating point values using ALP compression
  ///
  /// Samples the input data and selects the best compression mode (ALP or
  /// ALP-RD) based on estimated compression ratios. The mode can be overridden
  /// using the enforce_mode parameter.
  ///
  /// \param[in] decomp pointer to the input that is to be encoded
  /// \param[in] decomp_size size of decomp in bytes.
  ///            This needs to be a multiple of sizeof(T).
  /// \param[out] comp pointer to the memory region we will encode into.
  ///             The caller is responsible for ensuring this is big enough.
  /// \param[in,out] comp_size the actual size of the encoded data in bytes,
  ///                expects the size of comp as input. If this is too small,
  ///                this is set to 0 and we bail out.
  /// \param[in] enforce_mode if provided, forces the specified compression mode.
  ///            If nullopt, the mode is automatically selected based on
  ///            estimated compression ratios.
  static void Encode(const T* decomp, size_t decomp_size, char* comp,
                     size_t* comp_size,
                     std::optional<AlpMode> enforce_mode = std::nullopt);

  /// \brief Encode floating point values using a pre-computed sampling preset
  ///
  /// Uses the provided preset to encode data, avoiding the overhead of
  /// re-sampling. This is useful when encoding multiple data blocks that
  /// share similar characteristics (e.g., same column in multiple pages).
  ///
  /// \param[in] decomp pointer to the input that is to be encoded
  /// \param[in] decomp_size size of decomp in bytes.
  ///            This needs to be a multiple of sizeof(T).
  /// \param[out] comp pointer to the memory region we will encode into.
  ///             The caller is responsible for ensuring this is big enough.
  /// \param[in,out] comp_size the actual size of the encoded data in bytes,
  ///                expects the size of comp as input. If this is too small,
  ///                this is set to 0 and we bail out.
  /// \param[in] preset pre-computed AlpSamplerResult from CreateSamplingPreset()
  ///            or a previous Encode call. Contains the recommended mode
  ///            (ALP/ALP-RD) and the corresponding preset parameters.
  static void EncodeWithPreset(const T* decomp, size_t decomp_size, char* comp,
                               size_t* comp_size, const AlpSamplerResult& preset);

  /// \brief Decode floating point values
  ///
  /// Automatically detects the compression mode from the header and applies
  /// the appropriate decompression algorithm.
  ///
  /// \param[out] decomp pointer to the memory region we will decode into.
  ///             The caller is responsible for ensuring this is big enough
  ///             to hold num_elements values.
  /// \param[in] num_elements number of elements to decode (from page header).
  ///            Uses uint32_t since Parquet page headers use i32 for num_values.
  /// \param[in] comp pointer to the input that is to be decoded
  /// \param[in] comp_size size of the input in bytes (from page header)
  /// \tparam TargetType the type that is used to store the output.
  ///         May not be a narrowing conversion from T.
  template <typename TargetType>
  static void Decode(TargetType* decomp, uint32_t num_elements, const char* comp,
                     size_t comp_size);

  /// \brief Get the maximum compressed size of an uncompressed buffer
  ///
  /// Returns the maximum possible compressed size assuming worst case for
  /// both ALP and ALP-RD modes.
  ///
  /// \param[in] decomp_size the size of the uncompressed buffer in bytes
  /// \return the maximum size of the compressed buffer
  static uint64_t GetMaxCompressedSize(uint64_t decomp_size);

 private:
  struct AlpHeader;

  /// \brief Tracks the progress of a compression operation
  ///
  /// Used to report how much data was consumed and produced during encoding.
  struct CompressionProgress {
    /// Number of compressed bytes written to output
    uint64_t num_compressed_bytes_produced = 0;
    /// Number of input elements consumed
    uint64_t num_uncompressed_elements_taken = 0;
  };

  /// \brief Tracks the progress of a decompression operation
  ///
  /// Used to report how much data was consumed and produced during decoding.
  struct DecompressionProgress {
    /// Number of decompressed elements written
    uint64_t num_decompressed_elements_produced = 0;
    /// Number of compressed bytes consumed
    uint64_t num_compressed_bytes_taken = 0;
  };

  /// \brief Compress a buffer using the ALP decimal compression
  ///
  /// \param[in] decomp array of floating point numbers to compress
  /// \param[in] element_count the number of floating point numbers
  /// \param[out] comp the buffer to be compressed into
  /// \param[in] comp_size the size of the compression buffer
  /// \param[in] combinations the encoding preset to use
  /// \return the compression progress
  static CompressionProgress EncodeAlp(const T* decomp, uint64_t element_count,
                                       char* comp, size_t comp_size,
                                       const AlpEncodingPreset& combinations);

  /// \brief Compress a buffer using the ALP-RD real-doubles compression
  ///
  /// \param[in] decomp array of floating point numbers to compress
  /// \param[in] element_count the number of floating point numbers
  /// \param[out] comp the buffer to be compressed into
  /// \param[in] comp_size the size of the compression buffer
  /// \param[in] preset the ALP-RD encoding preset to use
  /// \return the compression progress
  static CompressionProgress EncodeAlpRd(const T* decomp, uint64_t element_count,
                                         char* comp, size_t comp_size,
                                         const AlpRdEncodingPreset& preset);

  /// \brief Decompress a buffer using the ALP decimal compression
  ///
  /// \param[out] decomp the buffer to be decompressed into
  /// \param[in] decomp_element_count the number of floats to decompress
  /// \param[in] comp the compressed buffer to be decompressed
  /// \param[in] comp_size the size of the compressed data
  /// \param[in] integer_encoding the bit packing layout used
  /// \param[in] vector_size the number of elements per vector (from header)
  /// \param[in] total_elements the total number of elements in the page (from header).
  ///            Uses uint32_t since Parquet page headers use i32 for num_values.
  /// \return the decompression progress
  /// \tparam TargetType the type that is used to store the output.
  ///         May not be a narrowing conversion from T.
  template <typename TargetType>
  static DecompressionProgress DecodeAlp(TargetType* decomp, size_t decomp_element_count,
                                         const char* comp, size_t comp_size,
                                         AlpIntegerEncoding integer_encoding,
                                         uint32_t vector_size, uint32_t total_elements);

  /// \brief Decompress a buffer using the ALP-RD real-doubles compression
  ///
  /// \param[out] decomp the buffer to be decompressed into
  /// \param[in] decomp_element_count the number of floats to decompress
  /// \param[in] comp the compressed buffer to be decompressed
  /// \param[in] comp_size the size of the compressed data
  /// \param[in] settings the ALP-RD encoding settings (dictionary)
  /// \param[in] vector_size the number of elements per vector (from header)
  /// \param[in] total_elements the total number of elements in the page (from header).
  /// \return the decompression progress
  /// \tparam TargetType the type that is used to store the output.
  template <typename TargetType>
  static DecompressionProgress DecodeAlpRd(TargetType* decomp,
                                           size_t decomp_element_count,
                                           const char* comp, size_t comp_size,
                                           const AlpRdEncodingSettings& settings,
                                           uint32_t vector_size,
                                           uint32_t total_elements);

  /// \brief Load the AlpHeader from compressed data
  ///
  /// \param[in] comp the compressed buffer
  /// \param[in] comp_size the size of the compressed data
  /// \return the AlpHeader from comp
  static AlpHeader LoadHeader(const char* comp, size_t comp_size);
};

}  // namespace alp
}  // namespace util
}  // namespace arrow
