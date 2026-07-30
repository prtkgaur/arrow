// Licensed to the Apache Software Foundation (ASF) under one or more
// contributor license agreements. See the NOTICE file distributed with this
// work for additional information regarding copyright ownership. The ASF
// licenses this file to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

// Standalone C++ comparison of FSST (Arrow PR #48232 vendored codec), a C++
// OnPair implementation, zstd level 1 and lz4, in one process on identical
// string corpora. Reports compression ratio and encode/decode throughput.
// Corpora are produced by the Rust bench-fsst-onpair/ generator (--dump-corpora).
//
// All codecs run single-threaded; pin the process with `taskset -c 0`, and run
// it twice using the warm second run.
//
// Build (from the Arrow repo root), one line:
//   g++ -std=c++17 -O3 -march=native -Icpp/src -Icpp/thirdparty/fsst
//   cpp/thirdparty/fsst/libfsst.cpp cpp/thirdparty/fsst/fsst_avx512.cpp
//   cpp/src/parquet/onpair/onpair.cc cpp/src/parquet/onpair/fsst_onpair_benchmark.cc
//   /usr/lib64/libzstd.so.1 /usr/lib64/liblz4.so.1 -o /tmp/fsst_onpair_bench
//
// Run:  taskset -c 0 /tmp/fsst_onpair_bench <corpora_dir>
//   (corpora_dir defaults to $ONPAIR_BENCH_DIR, then ./bench-fsst-onpair/corpora)

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "fsst.h"
#include "parquet/onpair/onpair.h"

// zstd: declare the small, ABI-stable subset we use so the standalone build
// needs only the installed libzstd (no dev header). Link libzstd.so directly.
extern "C" {
size_t ZSTD_compress(void* dst, size_t dstCapacity, const void* src, size_t srcSize, int level);
size_t ZSTD_decompress(void* dst, size_t dstCapacity, const void* src, size_t compressedSize);
size_t ZSTD_compressBound(size_t srcSize);
unsigned ZSTD_isError(size_t code);
// lz4 (fast block compressor) - ABI-stable subset; link the installed liblz4.
int LZ4_compressBound(int inputSize);
int LZ4_compress_default(const char* src, char* dst, int srcSize, int dstCapacity);
int LZ4_decompress_safe(const char* src, char* dst, int compressedSize, int dstCapacity);
}

namespace op = parquet::onpair;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int kEncodeIters = 3;
constexpr int kDecodeIters = 10;

double Mib(size_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// A packed corpus: concatenated bytes + (n+1) u32 offsets.
// Bits to store a value in [0, x]  (x==0 -> 0 bits).
inline size_t BitWidth(uint64_t x) {
  return x == 0 ? 0 : 64 - static_cast<size_t>(__builtin_clzll(x));
}
// Bits to index `count` distinct symbols [0, count)  (== ceil(log2 count), >=1).
inline size_t IndexBits(size_t count) {
  return count <= 1 ? 1 : (64 - static_cast<size_t>(__builtin_clzll(static_cast<uint64_t>(count - 1))));
}
inline size_t BitPackedBytes(size_t n, size_t bits) { return (n * bits + 7) / 8; }

struct Corpus {
  std::string name;
  std::vector<uint8_t> bytes;
  std::vector<uint32_t> offsets;
  size_t n_rows() const { return offsets.size() - 1; }
  size_t raw_bytes() const { return bytes.size(); }
  size_t max_row_len() const {
    size_t m = 0;
    for (size_t i = 0; i < n_rows(); ++i) m = std::max<size_t>(m, offsets[i + 1] - offsets[i]);
    return m;
  }
  // Realistic per-row row-length side array (delta offsets), bit-packed at the
  // width of the longest row. Charged to every value-preserving codec (FSST,
  // zstd, lz4, OnPair) so the row boundaries are accounted the way a real
  // columnar format stores them - not as raw (n+1) u32.
  size_t len_array_bytes() const {
    return BitPackedBytes(n_rows(), std::max<size_t>(1, BitWidth(max_row_len())));
  }
};

// Read a newline-delimited file (one row per line) into a packed corpus.
Corpus ReadCorpus(const std::filesystem::path& path) {
  Corpus c;
  c.name = path.stem().string();
  std::ifstream in(path, std::ios::binary);
  c.offsets.push_back(0);
  std::string line;
  while (std::getline(in, line)) {
    c.bytes.insert(c.bytes.end(), line.begin(), line.end());
    c.offsets.push_back(static_cast<uint32_t>(c.bytes.size()));
  }
  return c;
}

struct Measured {
  std::string label;
  size_t compressed_bytes = 0;
  double encode_mibs = 0;
  double decode_mibs = 0;
};

// FSST

// Compress the whole corpus into one packed buffer; returns compressed bytes,
// per-row lengths, and the serialized symbol-table size.
struct FsstEncoded {
  std::vector<uint8_t> output;   // packed compressed bytes
  size_t total = 0;              // used bytes in `output`
  size_t table_bytes = 0;        // fsst_export size (symbol table)
};

FsstEncoded FsstEncode(const Corpus& c) {
  size_t n = c.n_rows();
  std::vector<size_t> lenIn(n);
  std::vector<const unsigned char*> strIn(n);
  for (size_t i = 0; i < n; ++i) {
    lenIn[i] = c.offsets[i + 1] - c.offsets[i];
    strIn[i] = c.bytes.data() + c.offsets[i];
  }
  fsst_encoder_t* enc = fsst_create(n, lenIn.data(), strIn.data(), 0);

  // Conservative per-string bound from fsst.h: 7 + 2*len.
  size_t out_cap = 7 * n + 2 * c.raw_bytes() + 16;
  FsstEncoded e;
  e.output.resize(out_cap);
  std::vector<size_t> lenOut(n);
  std::vector<unsigned char*> strOut(n);
  size_t done = fsst_compress(enc, n, lenIn.data(), strIn.data(), out_cap, e.output.data(),
                              lenOut.data(), strOut.data());
  if (done != n) {
    std::fprintf(stderr, "FSST: only compressed %zu/%zu rows\n", done, n);
    std::abort();
  }
  e.total = 0;
  for (size_t i = 0; i < n; ++i) e.total += lenOut[i];

  unsigned char table[FSST_MAXHEADER];
  e.table_bytes = fsst_export(enc, table);
  fsst_destroy(enc);
  return e;
}

Measured RunFsst(const Corpus& c) {
  size_t n = c.n_rows();
  FsstEncoded e = FsstEncode(c);
  Measured m;
  m.label = "FSST";
  m.compressed_bytes = e.table_bytes + e.total + c.len_array_bytes();

  // Encode throughput (train + compress).
  std::vector<double> enc;
  for (int it = 0; it < kEncodeIters; ++it) {
    auto t0 = Clock::now();
    FsstEncoded tmp = FsstEncode(c);
    double dt = std::chrono::duration<double>(Clock::now() - t0).count();
    asm volatile("" ::"r"(tmp.total) : "memory");
    enc.push_back(Mib(c.raw_bytes()) / dt);
  }
  m.encode_mibs = Median(std::move(enc));

  // Rebuild a decoder from the packed stream for decode timing.
  std::vector<size_t> lenIn(n);
  std::vector<const unsigned char*> strIn(n);
  for (size_t i = 0; i < n; ++i) {
    lenIn[i] = c.offsets[i + 1] - c.offsets[i];
    strIn[i] = c.bytes.data() + c.offsets[i];
  }
  fsst_encoder_t* enc2 = fsst_create(n, lenIn.data(), strIn.data(), 0);
  fsst_decoder_t dec = fsst_decoder(enc2);
  fsst_destroy(enc2);

  size_t cap = c.raw_bytes() + 16;
  // Correctness: decoding the whole packed stream reconstructs the input.
  {
    std::vector<uint8_t> out(cap);
    size_t w = fsst_decompress(&dec, e.total, e.output.data(), cap, out.data());
    if (w != c.raw_bytes() || std::memcmp(out.data(), c.bytes.data(), c.raw_bytes()) != 0) {
      std::fprintf(stderr, "FSST roundtrip mismatch on %s (w=%zu raw=%zu)\n", c.name.c_str(), w,
                   c.raw_bytes());
      std::abort();
    }
  }
  std::vector<double> dec_r;
  for (int it = 0; it < kDecodeIters; ++it) {
    std::vector<uint8_t> out(cap);
    auto t0 = Clock::now();
    size_t w = fsst_decompress(&dec, e.total, e.output.data(), cap, out.data());
    double dt = std::chrono::duration<double>(Clock::now() - t0).count();
    asm volatile("" ::"r"(w) : "memory");
    dec_r.push_back(Mib(c.raw_bytes()) / dt);
  }
  m.decode_mibs = Median(std::move(dec_r));
  return m;
}

// zstd (block-compression baseline; no random access)

// Compresses the whole concatenated corpus in one frame at the given level.
// Unlike FSST/OnPair this gives no per-row random access - it's a reference for
// what a general-purpose block compressor achieves on the same bytes.
Measured RunZstd(const Corpus& c, int level) {
  size_t bound = ZSTD_compressBound(c.raw_bytes());
  std::vector<uint8_t> comp(bound);
  size_t csize = ZSTD_compress(comp.data(), bound, c.bytes.data(), c.raw_bytes(), level);
  if (ZSTD_isError(csize)) {
    std::fprintf(stderr, "zstd compress error on %s\n", c.name.c_str());
    std::abort();
  }
  Measured m;
  m.label = "zstd(" + std::to_string(level) + ")";
  // Add (n+1) u32 row offsets so row recovery is accounted for, as with the
  // other codecs (zstd's frame decompresses to concatenated plaintext only).
  m.compressed_bytes = csize + c.len_array_bytes();

  std::vector<double> enc;
  for (int it = 0; it < kEncodeIters; ++it) {
    auto t0 = Clock::now();
    size_t r = ZSTD_compress(comp.data(), bound, c.bytes.data(), c.raw_bytes(), level);
    double dt = std::chrono::duration<double>(Clock::now() - t0).count();
    asm volatile("" ::"r"(r) : "memory");
    enc.push_back(Mib(c.raw_bytes()) / dt);
  }
  m.encode_mibs = Median(std::move(enc));

  size_t cap = c.raw_bytes() + 16;
  {
    std::vector<uint8_t> out(cap);
    size_t w = ZSTD_decompress(out.data(), cap, comp.data(), csize);
    if (ZSTD_isError(w) || w != c.raw_bytes() ||
        std::memcmp(out.data(), c.bytes.data(), c.raw_bytes()) != 0) {
      std::fprintf(stderr, "zstd roundtrip mismatch on %s\n", c.name.c_str());
      std::abort();
    }
  }
  std::vector<double> dec_r;
  for (int it = 0; it < kDecodeIters; ++it) {
    std::vector<uint8_t> out(cap);
    auto t0 = Clock::now();
    size_t w = ZSTD_decompress(out.data(), cap, comp.data(), csize);
    double dt = std::chrono::duration<double>(Clock::now() - t0).count();
    asm volatile("" ::"r"(w) : "memory");
    dec_r.push_back(Mib(c.raw_bytes()) / dt);
  }
  m.decode_mibs = Median(std::move(dec_r));
  return m;
}

// lz4 (fast block-compression baseline; no random access)

Measured RunLz4(const Corpus& c) {
  int raw = static_cast<int>(c.raw_bytes());
  int bound = LZ4_compressBound(raw);
  std::vector<char> comp(bound);
  int csize = LZ4_compress_default(reinterpret_cast<const char*>(c.bytes.data()), comp.data(), raw,
                                   bound);
  if (csize <= 0) {
    std::fprintf(stderr, "lz4 compress error on %s\n", c.name.c_str());
    std::abort();
  }
  Measured m;
  m.label = "lz4";
  m.compressed_bytes = static_cast<size_t>(csize) + c.len_array_bytes();  // + bit-packed lengths

  std::vector<double> enc;
  for (int it = 0; it < kEncodeIters; ++it) {
    auto t0 = Clock::now();
    int r = LZ4_compress_default(reinterpret_cast<const char*>(c.bytes.data()), comp.data(), raw,
                                 bound);
    double dt = std::chrono::duration<double>(Clock::now() - t0).count();
    asm volatile("" ::"r"(r) : "memory");
    enc.push_back(Mib(c.raw_bytes()) / dt);
  }
  m.encode_mibs = Median(std::move(enc));

  int cap = raw + 16;
  {
    std::vector<char> out(cap);
    int w = LZ4_decompress_safe(comp.data(), out.data(), csize, cap);
    if (w != raw || std::memcmp(out.data(), c.bytes.data(), c.raw_bytes()) != 0) {
      std::fprintf(stderr, "lz4 roundtrip mismatch on %s\n", c.name.c_str());
      std::abort();
    }
  }
  std::vector<double> dec_r;
  for (int it = 0; it < kDecodeIters; ++it) {
    std::vector<char> out(cap);
    auto t0 = Clock::now();
    int w = LZ4_decompress_safe(comp.data(), out.data(), csize, cap);
    double dt = std::chrono::duration<double>(Clock::now() - t0).count();
    asm volatile("" ::"r"(w) : "memory");
    dec_r.push_back(Mib(c.raw_bytes()) / dt);
  }
  m.decode_mibs = Median(std::move(dec_r));
  return m;
}

// OnPair

Measured RunOnPair(const Corpus& c, uint8_t bits, double threshold) {
  op::Config cfg;
  cfg.max_dict_bits = bits;
  cfg.threshold_fraction = threshold;
  cfg.seed = 42;
  size_t n = c.n_rows();

  op::Column col = op::Compress(c.bytes.data(), c.raw_bytes(), c.offsets.data(), n, cfg);
  Measured m;
  m.label = "OnPair" + std::to_string(bits);
  // Realistic bit-packed accounting: codes packed at the true code width for the
  // trained dictionary (not a fixed u16), dictionary offsets bit-packed, and the
  // shared per-row length array (in place of the OnPair code-offset array).
  size_t dict_bytes = col.dict.logical_bytes();
  size_t code_bits = IndexBits(col.dict.num_tokens());
  size_t codes = BitPackedBytes(col.codes.size(), code_bits);
  size_t dict_offsets =
      BitPackedBytes(col.dict.offsets.size(), std::max<size_t>(1, BitWidth(dict_bytes)));
  m.compressed_bytes = dict_bytes + dict_offsets + codes + c.len_array_bytes();

  std::vector<double> enc;
  for (int it = 0; it < kEncodeIters; ++it) {
    auto t0 = Clock::now();
    op::Column tmp = op::Compress(c.bytes.data(), c.raw_bytes(), c.offsets.data(), n, cfg);
    double dt = std::chrono::duration<double>(Clock::now() - t0).count();
    asm volatile("" ::"r"(tmp.codes.size()) : "memory");
    enc.push_back(Mib(c.raw_bytes()) / dt);
  }
  m.encode_mibs = Median(std::move(enc));

  // Decode the *packed* code stream (unpack `code_bits` per code + gather), so
  // decode pays the real bit-unpacking cost that the packed ratio implies.
  size_t cap = op::DecodedLen(col) + op::kDecodePadding;
  std::vector<uint32_t> cw(col.codes.begin(), col.codes.end());
  std::vector<uint8_t> packed = op::PackValues(cw.data(), cw.size(), code_bits);
  {
    std::vector<uint8_t> out(cap, 0);
    size_t w = op::DecompressPacked(col.dict, packed.data(), col.codes.size(), code_bits, out.data());
    if (w != c.raw_bytes() || std::memcmp(out.data(), c.bytes.data(), c.raw_bytes()) != 0) {
      std::fprintf(stderr, "OnPair%u packed roundtrip mismatch on %s (w=%zu raw=%zu)\n", bits,
                   c.name.c_str(), w, c.raw_bytes());
      std::abort();
    }
  }
  std::vector<double> dec_r;
  for (int it = 0; it < kDecodeIters; ++it) {
    std::vector<uint8_t> out(cap, 0);
    auto t0 = Clock::now();
    size_t w = op::DecompressPacked(col.dict, packed.data(), col.codes.size(), code_bits, out.data());
    double dt = std::chrono::duration<double>(Clock::now() - t0).count();
    asm volatile("" ::"r"(w) : "memory");
    dec_r.push_back(Mib(c.raw_bytes()) / dt);
  }
  m.decode_mibs = Median(std::move(dec_r));
  return m;
}

// Bit-packed size of an OnPair column (dict + bit-packed dict offsets + codes at
// the true code width + the shared per-row length array).
size_t OnPairSize(const op::Column& col, const Corpus& c) {
  size_t db = col.dict.logical_bytes();
  return db + BitPackedBytes(col.dict.offsets.size(), std::max<size_t>(1, BitWidth(db))) +
         BitPackedBytes(col.codes.size(), IndexBits(col.dict.num_tokens())) + c.len_array_bytes();
}

// OnPair with the dictionary bit-width chosen per column: try 9..16 and keep the
// width that minimizes bit-packed size, then report that width's ratio/decode.
// (A production encoder would pick the width from one training pass via the
// token-gain curve; here we measure the achievable optimum directly.)
Measured RunOnPairAuto(const Corpus& c, double threshold) {
  size_t n = c.n_rows();
  uint8_t best_bits = 9;
  size_t best_sz = SIZE_MAX;
  for (uint8_t b = 9; b <= 16; ++b) {
    op::Config cfg{b, threshold, 42};
    op::Column col = op::Compress(c.bytes.data(), c.raw_bytes(), c.offsets.data(), n, cfg);
    size_t sz = OnPairSize(col, c);
    if (sz < best_sz) { best_sz = sz; best_bits = b; }
  }

  op::Config cfg{best_bits, threshold, 42};
  op::Column col = op::Compress(c.bytes.data(), c.raw_bytes(), c.offsets.data(), n, cfg);
  Measured m;
  // Report the *stored* code width = ceil(log2(tokens trained)), which is what
  // determines size. It can be < the budget when training saturates first.
  size_t stored_bits = IndexBits(col.dict.num_tokens());
  m.label = "OnPair-auto(" + std::to_string(stored_bits) + "b)";
  m.compressed_bytes = OnPairSize(col, c);

  // Encode throughput at the chosen width (a real encoder adds only a cheap
  // one-pass width estimate, not a full re-search, so this is representative).
  std::vector<double> enc;
  for (int it = 0; it < kEncodeIters; ++it) {
    auto t0 = Clock::now();
    op::Column tmp = op::Compress(c.bytes.data(), c.raw_bytes(), c.offsets.data(), n, cfg);
    double dt = std::chrono::duration<double>(Clock::now() - t0).count();
    asm volatile("" ::"r"(tmp.codes.size()) : "memory");
    enc.push_back(Mib(c.raw_bytes()) / dt);
  }
  m.encode_mibs = Median(std::move(enc));

  // Packed decode at the chosen stored width.
  size_t cap = op::DecodedLen(col) + op::kDecodePadding;
  std::vector<uint32_t> cw(col.codes.begin(), col.codes.end());
  std::vector<uint8_t> packed = op::PackValues(cw.data(), cw.size(), stored_bits);
  std::vector<double> dec_r;
  for (int it = 0; it < kDecodeIters; ++it) {
    std::vector<uint8_t> out(cap, 0);
    auto t0 = Clock::now();
    size_t w = op::DecompressPacked(col.dict, packed.data(), col.codes.size(), stored_bits, out.data());
    double dt = std::chrono::duration<double>(Clock::now() - t0).count();
    asm volatile("" ::"r"(w) : "memory");
    dec_r.push_back(Mib(c.raw_bytes()) / dt);
  }
  m.decode_mibs = Median(std::move(dec_r));
  return m;
}

// dedup-then-OnPair
//
// The layout a real columnar format uses for repetitive columns: encode the
// column as bit-packed references into the set of distinct values, and run
// OnPair over only those distinct values. Removes whole-value repetition (which
// OnPair's <=16-byte substring dictionary can't exploit) before compressing the
// residual substring redundancy. Matches the OnPair README's guidance for
// low-cardinality columns.

inline size_t CeilLog2(size_t x) {
  if (x <= 1) return 1;  // need >=1 bit even for a 2-value dictionary
  return 64 - static_cast<size_t>(__builtin_clzll(x - 1));
}

// Fast allocation-free byte-range hash, consuming 8 bytes per step (with a
// tail) and a final avalanche - far cheaper than a byte-at-a-time FNV for the
// short strings that dominate low-cardinality columns.
inline uint64_t HashBytes(const uint8_t* p, size_t len) {
  uint64_t h = 0x9E3779B97F4A7C15ull ^ (static_cast<uint64_t>(len) * 0xff51afd7ed558ccdull);
  size_t i = 0;
  for (; i + 8 <= len; i += 8) {
    uint64_t w;
    std::memcpy(&w, p + i, 8);
    h = (h ^ w) * 0x100000001b3ull;
  }
  if (i < len) {
    uint64_t w = 0;
    std::memcpy(&w, p + i, len - i);
    h = (h ^ w) * 0x100000001b3ull;
  }
  h ^= h >> 29;
  h *= 0xbf58476d1ce4e5b9ull;
  h ^= h >> 32;
  return h;
}

Measured RunOnPairDedup(const Corpus& c, uint8_t bits, double threshold) {
  op::Config cfg;
  cfg.max_dict_bits = bits;
  cfg.threshold_fraction = threshold;
  cfg.seed = 42;
  size_t n = c.n_rows();

  // Build the distinct-value set in first-seen order + per-row references.
  // Open-addressing (linear-probe) table keyed on the row bytes, assigning ids
  // in first-seen order - same distinct set/order as a std::unordered_map would
  // give (so ratios are identical) but without per-key node allocation or the
  // std::hash<string_view> + pointer-chase overhead, which dominated encode.
  auto build_dedup = [&](std::vector<uint8_t>* d_bytes, std::vector<uint32_t>* d_offsets,
                         std::vector<uint32_t>* refs) {
    size_t cap = 1;
    while (cap < n * 2) cap <<= 1;  // power-of-two, <=50% load
    const uint32_t kEmpty = 0xFFFFFFFFu;
    std::vector<uint32_t> table(cap, kEmpty);  // slot -> distinct id
    uint64_t mask = cap - 1;
    d_offsets->push_back(0);
    refs->resize(n);
    uint32_t n_distinct = 0;
    for (size_t i = 0; i < n; ++i) {
      const uint8_t* row = c.bytes.data() + c.offsets[i];
      size_t len = c.offsets[i + 1] - c.offsets[i];
      uint64_t slot = HashBytes(row, len) & mask;
      uint32_t id;
      for (;;) {
        uint32_t cur = table[slot];
        if (cur == kEmpty) {  // new distinct value
          id = n_distinct++;
          table[slot] = id;
          d_bytes->insert(d_bytes->end(), row, row + len);
          d_offsets->push_back(static_cast<uint32_t>(d_bytes->size()));
          break;
        }
        size_t off = (*d_offsets)[cur];
        size_t clen = (*d_offsets)[cur + 1] - off;
        if (clen == len && std::memcmp(d_bytes->data() + off, row, len) == 0) {
          id = cur;  // seen before
          break;
        }
        slot = (slot + 1) & mask;  // linear probe
      }
      (*refs)[i] = id;
    }
    return static_cast<size_t>(n_distinct);
  };

  std::vector<uint8_t> d_bytes;
  std::vector<uint32_t> d_offsets;
  std::vector<uint32_t> refs;
  size_t n_distinct = build_dedup(&d_bytes, &d_offsets, &refs);

  op::Column col = op::Compress(d_bytes.data(), d_bytes.size(), d_offsets.data(), n_distinct, cfg);

  Measured m;
  m.label = "OnPair" + std::to_string(bits) + "-dedup";
  // Realistic bit-packed accounting, applied to the OnPair-encoded distinct set
  // (dict + true-width codes + bit-packed dict offsets + a distinct-value length
  // array) plus the bit-packed per-row reference (index) column.
  size_t dict_bytes = col.dict.logical_bytes();
  size_t code_bits = IndexBits(col.dict.num_tokens());
  size_t codes = BitPackedBytes(col.codes.size(), code_bits);
  size_t dict_offsets =
      BitPackedBytes(col.dict.offsets.size(), std::max<size_t>(1, BitWidth(dict_bytes)));
  size_t dmax = 0;
  for (size_t j = 0; j + 1 < d_offsets.size(); ++j)
    dmax = std::max<size_t>(dmax, d_offsets[j + 1] - d_offsets[j]);
  size_t distinct_len_bytes = BitPackedBytes(n_distinct, std::max<size_t>(1, BitWidth(dmax)));
  size_t onpair_bytes = dict_bytes + dict_offsets + codes + distinct_len_bytes;
  size_t refs_bytes = BitPackedBytes(n, IndexBits(n_distinct));  // index column
  m.compressed_bytes = onpair_bytes + refs_bytes;

  std::vector<double> enc;
  for (int it = 0; it < kEncodeIters; ++it) {
    std::vector<uint8_t> db;
    std::vector<uint32_t> doff;
    std::vector<uint32_t> rf;
    auto t0 = Clock::now();
    size_t nd = build_dedup(&db, &doff, &rf);
    op::Column tmp = op::Compress(db.data(), db.size(), doff.data(), nd, cfg);
    double dt = std::chrono::duration<double>(Clock::now() - t0).count();
    asm volatile("" ::"r"(tmp.codes.size()) : "memory");
    enc.push_back(Mib(c.raw_bytes()) / dt);
  }
  m.encode_mibs = Median(std::move(enc));

  // Decode: materialize distinct values once (OnPair decode), then gather rows
  // by reference. dbuf[d_offsets[id]..] holds distinct value `id` (decode
  // reproduces the concatenated distinct bytes in id order).
  //
  // Gather uses a branchless fixed-16-byte copy when the value is <=16 bytes
  // (the common case for the low-cardinality columns where dedup shines): one
  // 128-bit store instead of a variable-length memcpy dispatch. Safe because the
  // OnPair decode buffer is read-padded by kDecodePadding(16) and `out` carries
  // 16 bytes of write padding; the cursor advances by the true length so the
  // over-store is overwritten by the next row (or absorbed by the pad on the
  // last). Values >16 bytes fall back to an exact memcpy.
  size_t dlen = op::DecodedLen(col);
  size_t cap = c.raw_bytes() + 16;
  // Pack the distinct-set code stream and the per-row reference (index) column,
  // so decode pays the real unpacking cost the packed sizes imply.
  std::vector<uint32_t> cw(col.codes.begin(), col.codes.end());
  std::vector<uint8_t> packed_codes = op::PackValues(cw.data(), cw.size(), code_bits);
  size_t ref_bits = IndexBits(n_distinct);
  std::vector<uint8_t> packed_refs = op::PackValues(refs.data(), n, ref_bits);

  // Materialize the distinct values (unpack the OnPair code stream), then gather
  // each row by unpacking its reference and copying the referenced value. <=16-byte
  // values use one branchless 128-bit store (dbuf/out are 16-byte padded).
  auto decode = [&](uint8_t* dbuf, uint8_t* out) -> size_t {
    op::DecompressPacked(col.dict, packed_codes.data(), col.codes.size(), code_bits, dbuf);
    size_t w = 0, bp = 0;
    for (size_t i = 0; i < n; ++i) {
      uint32_t id = op::GetBits(packed_refs.data(), bp, ref_bits);
      bp += ref_bits;
      size_t off = d_offsets[id];
      size_t len = d_offsets[id + 1] - off;
      const uint8_t* src = dbuf + off;
      if (len <= 16) std::memcpy(out + w, src, 16);
      else std::memcpy(out + w, src, len);
      w += len;
    }
    return w;
  };

  {
    std::vector<uint8_t> dbuf(dlen + op::kDecodePadding, 0), out(cap);
    size_t w = decode(dbuf.data(), out.data());
    if (w != c.raw_bytes() || std::memcmp(out.data(), c.bytes.data(), c.raw_bytes()) != 0) {
      std::fprintf(stderr, "OnPair%u-dedup packed roundtrip mismatch on %s\n", bits, c.name.c_str());
      std::abort();
    }
  }
  std::vector<double> dec_r;
  for (int it = 0; it < kDecodeIters; ++it) {
    std::vector<uint8_t> dbuf(dlen + op::kDecodePadding, 0), out(cap);
    auto t0 = Clock::now();
    size_t w = decode(dbuf.data(), out.data());
    double dt = std::chrono::duration<double>(Clock::now() - t0).count();
    asm volatile("" ::"r"(w) : "memory");
    dec_r.push_back(Mib(c.raw_bytes()) / dt);
  }
  m.decode_mibs = Median(std::move(dec_r));
  return m;
}

// TPC-H columns train with a 0.2 sample fraction; the URL corpus with 0.5
// (matching the Rust harness).
double ThresholdFor(const std::string& name) {
  return name.rfind("tpch_", 0) == 0 ? 0.2 : 0.5;
}

}  // namespace

int main(int argc, char** argv) {
  std::string dir;
  if (argc > 1) {
    dir = argv[1];
  } else if (const char* env = std::getenv("ONPAIR_BENCH_DIR")) {
    dir = env;
  } else {
    dir = "bench-fsst-onpair/corpora";
  }

  std::vector<std::filesystem::path> files;
  for (const auto& e : std::filesystem::directory_iterator(dir)) {
    if (e.path().extension() == ".txt") files.push_back(e.path());
  }
  std::sort(files.begin(), files.end());
  if (files.empty()) {
    std::fprintf(stderr, "no .txt corpora in %s\n", dir.c_str());
    return 1;
  }

  std::printf("%-26s %10s %10s   %7s %9s %9s\n", "corpus", "rows", "raw MiB", "ratio",
              "enc MiB/s", "dec MiB/s");
  std::printf("%s\n", std::string(90, '-').c_str());

  for (const auto& path : files) {
    Corpus c = ReadCorpus(path);
    double threshold = ThresholdFor(c.name);

    Measured fsst = RunFsst(c);
    Measured zstd1 = RunZstd(c, 1);
    Measured lz4 = RunLz4(c);
    Measured op16 = RunOnPair(c, 16, threshold);
    Measured opauto = RunOnPairAuto(c, threshold);
    Measured op16d = RunOnPairDedup(c, 16, threshold);

    std::printf("%-26s %10zu %10.2f\n", c.name.c_str(), c.n_rows(), Mib(c.raw_bytes()));
    for (const Measured* m : {&fsst, &zstd1, &lz4, &op16, &opauto, &op16d}) {
      double ratio = static_cast<double>(c.raw_bytes()) / static_cast<double>(m->compressed_bytes);
      std::printf("  %-24s %10s %10.2f  %7.3fx %9.1f %9.1f\n", m->label.c_str(), "",
                  Mib(m->compressed_bytes), ratio, m->encode_mibs, m->decode_mibs);
    }
    double r_fsst = static_cast<double>(c.raw_bytes()) / fsst.compressed_bytes;
    double r_zstd = static_cast<double>(c.raw_bytes()) / zstd1.compressed_bytes;
    double r_op16 = static_cast<double>(c.raw_bytes()) / op16.compressed_bytes;
    std::printf("  -> OnPair16 vs FSST:    ratio %+.1f%%, encode %+.1f%%, decode %+.1f%%\n",
                (r_op16 / r_fsst - 1.0) * 100.0, (op16.encode_mibs / fsst.encode_mibs - 1.0) * 100.0,
                (op16.decode_mibs / fsst.decode_mibs - 1.0) * 100.0);
    std::printf("  -> OnPair16 vs zstd(1): ratio %+.1f%%, encode %+.1f%%, decode %+.1f%%\n",
                (r_op16 / r_zstd - 1.0) * 100.0,
                (op16.encode_mibs / zstd1.encode_mibs - 1.0) * 100.0,
                (op16.decode_mibs / zstd1.decode_mibs - 1.0) * 100.0);
    double r_op16d = static_cast<double>(c.raw_bytes()) / op16d.compressed_bytes;
    std::printf("  -> OnPair16-dedup vs zstd(1): ratio %+.1f%%, decode %+.1f%%\n\n",
                (r_op16d / r_zstd - 1.0) * 100.0,
                (op16d.decode_mibs / zstd1.decode_mibs - 1.0) * 100.0);
  }
  return 0;
}
