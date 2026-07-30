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

// Visible round-trip proof for the OnPair port: decode both plain OnPair16 and
// OnPair16-dedup and check EVERY row equals the original bytes, then print a few
// concrete original -> decoded samples so a human can eyeball the recovery.
//
// Build (one line): g++ -std=c++17 -O2 -Icpp/src
//   cpp/src/parquet/onpair/onpair.cc cpp/src/parquet/onpair/verify_roundtrip.cc -o /tmp/verify
// Run:   /tmp/verify bench-fsst-onpair/corpora/tpch_l_shipmode.txt [more files...]

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "parquet/onpair/onpair.h"

namespace op = parquet::onpair;

namespace {

struct Corpus {
  std::vector<uint8_t> bytes;
  std::vector<uint32_t> offsets;
  size_t rows() const { return offsets.size() - 1; }
  std::string row(size_t i) const {
    return std::string(reinterpret_cast<const char*>(bytes.data() + offsets[i]),
                       offsets[i + 1] - offsets[i]);
  }
};

Corpus Read(const char* path) {
  Corpus c;
  std::ifstream in(path, std::ios::binary);
  c.offsets.push_back(0);
  std::string line;
  while (std::getline(in, line)) {
    c.bytes.insert(c.bytes.end(), line.begin(), line.end());
    c.offsets.push_back(static_cast<uint32_t>(c.bytes.size()));
  }
  return c;
}

// Compare a decoded concatenated buffer to the original, row by row.
// Returns the number of mismatching rows and the first mismatching index.
size_t CheckPerRow(const Corpus& c, const uint8_t* decoded, size_t dn, long* first_bad) {
  *first_bad = -1;
  if (dn != c.bytes.size()) {
    *first_bad = 0;
    return c.rows();
  }
  size_t bad = 0;
  for (size_t i = 0; i < c.rows(); ++i) {
    size_t off = c.offsets[i], len = c.offsets[i + 1] - off;
    if (std::memcmp(decoded + off, c.bytes.data() + off, len) != 0) {
      if (*first_bad < 0) *first_bad = static_cast<long>(i);
      ++bad;
    }
  }
  return bad;
}

std::string Trunc(const std::string& s, size_t n = 42) {
  return s.size() <= n ? s : s.substr(0, n) + "…";
}

void Samples(const Corpus& c, const uint8_t* decoded) {
  size_t r = c.rows();
  size_t idx[4] = {0, r / 3, (2 * r) / 3, r - 1};
  for (size_t k = 0; k < 4; ++k) {
    size_t i = idx[k];
    size_t off = c.offsets[i], len = c.offsets[i + 1] - off;
    std::string dec(reinterpret_cast<const char*>(decoded + off), len);
    std::string orig = c.row(i);
    std::printf("      row %-8zu original=%-44s decoded=%-44s %s\n", i,
                ("\"" + Trunc(orig) + "\"").c_str(), ("\"" + Trunc(dec) + "\"").c_str(),
                orig == dec ? "MATCH" : "*** MISMATCH ***");
  }
}

// OnPair16 (no dedup): compress then whole-column decode.
void VerifyOnPair(const Corpus& c) {
  op::Config cfg = op::Config::Dict16();
  cfg.threshold_fraction = 0.5;
  op::Column col = op::Compress(c.bytes.data(), c.bytes.size(), c.offsets.data(), c.rows(), cfg);
  std::vector<uint8_t> out(op::DecodedLen(col) + op::kDecodePadding, 0);
  size_t dn = op::DecompressInto(col, out.data());
  long bad_at;
  size_t bad = CheckPerRow(c, out.data(), dn, &bad_at);
  std::printf("    OnPair16       : %zu/%zu rows exact  %s\n", c.rows() - bad, c.rows(),
              bad == 0 ? "[OK]" : "[FAIL]");
  Samples(c, out.data());
}

// OnPair16-dedup: dictionary-encode, OnPair the distinct values, then gather.
void VerifyOnPairDedup(const Corpus& c) {
  size_t n = c.rows();
  std::unordered_map<std::string, uint32_t> ids;
  std::vector<uint8_t> d_bytes;
  std::vector<uint32_t> d_offsets{0};
  std::vector<uint32_t> refs(n);
  for (size_t i = 0; i < n; ++i) {
    std::string v = c.row(i);
    auto it = ids.find(v);
    uint32_t id;
    if (it == ids.end()) {
      id = static_cast<uint32_t>(ids.size());
      d_bytes.insert(d_bytes.end(), v.begin(), v.end());
      d_offsets.push_back(static_cast<uint32_t>(d_bytes.size()));
      ids.emplace(std::move(v), id);
    } else {
      id = it->second;
    }
    refs[i] = id;
  }
  op::Config cfg = op::Config::Dict16();
  cfg.threshold_fraction = 0.5;
  op::Column col =
      op::Compress(d_bytes.data(), d_bytes.size(), d_offsets.data(), ids.size(), cfg);
  std::vector<uint8_t> dbuf(op::DecodedLen(col) + op::kDecodePadding, 0);
  op::DecompressInto(col, dbuf.data());  // distinct values, concatenated in id order
  std::vector<uint8_t> out(c.bytes.size() + 16, 0);
  size_t w = 0;
  for (size_t i = 0; i < n; ++i) {
    uint32_t id = refs[i];
    size_t off = d_offsets[id], len = d_offsets[id + 1] - off;
    std::memcpy(out.data() + w, dbuf.data() + off, len);
    w += len;
  }
  long bad_at;
  size_t bad = CheckPerRow(c, out.data(), w, &bad_at);
  std::printf("    OnPair16-dedup : %zu/%zu rows exact  (%zu distinct)  %s\n", n - bad, n,
              ids.size(), bad == 0 ? "[OK]" : "[FAIL]");
  Samples(c, out.data());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <corpus.txt> [more...]\n", argv[0]);
    return 2;
  }
  int failures = 0;
  for (int a = 1; a < argc; ++a) {
    Corpus c = Read(argv[a]);
    std::printf("\n%s  (%zu rows, %.2f MiB)\n", argv[a], c.rows(),
                c.bytes.size() / (1024.0 * 1024.0));
    VerifyOnPair(c);
    VerifyOnPairDedup(c);
  }
  (void)failures;
  return 0;
}
