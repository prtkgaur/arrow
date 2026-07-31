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

#include "parquet/onpair/onpair.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <unordered_map>

namespace parquet::onpair {
namespace {

constexpr size_t kBucketPrefixLen = 8;
constexpr size_t kPromoteThreshold = 128;

// Little-endian packing helpers

/// Pack the low min(len, data_len, 8) bytes of `data` into a little-endian u64;
/// higher bytes read as zero.
inline uint64_t LoadLeU64(const uint8_t* data, size_t data_len, size_t len) {
  size_t n = (len >= kBucketPrefixLen && data_len >= kBucketPrefixLen)
                 ? kBucketPrefixLen
                 : std::min(len, data_len);
  uint64_t v = 0;
  std::memcpy(&v, data, n);  // little-endian host
  return v;
}

/// Mask of the low len*8 bits in a u64.
inline uint64_t MaskU64(size_t len) {
  return len >= 8 ? ~uint64_t{0} : ((uint64_t{1} << (len * 8)) - 1);
}

/// Count of matching low bytes between two packed suffixes.
inline size_t MatchingLowBytes(uint64_t x) {
  return x == 0 ? 8 : (static_cast<size_t>(__builtin_ctzll(x)) >> 3);
}

// Longest-prefix matcher
// Two-tier index per the paper (sec 3.4.1): a hash map for tokens <=8 bytes, and
// 8-byte-prefix buckets (suffixes sorted descending) for 9..16-byte tokens.
// DEVIATION FROM PAPER (D3): the paper's OnPair16 caps each long bucket at 128
// suffixes (sec 3.4.4, dropping extras); this port instead promotes an over-full
// bucket to a trie (PROMOTE_THRESHOLD), keeping all suffixes. Also, the paper's
// static parsing phase (sec 3.4.3) finalizes long-pattern lookup with a minimal
// perfect hash; this port keeps std::unordered_map (the paper notes the
// perfect-hash path is Rust-only). Encode-time behavior only.

struct LongEntry {
  uint64_t suffix;
  uint8_t slen;
  Token token;
};

struct TrieNode {
  int token = -1;  // -1 == none
  std::vector<std::pair<uint8_t, uint32_t>> children;
};

struct Bucket {
  std::vector<LongEntry> entries;
  int32_t trie_root = -1;  // >=0 once promoted
};

class LongestPrefixMatcher {
 public:
  /// Empty matcher pre-loaded with the 256 single-byte tokens (ids 0..255).
  static LongestPrefixMatcher New() {
    LongestPrefixMatcher m;
    for (uint16_t i = 0; i <= 255; ++i) {
      uint8_t b = static_cast<uint8_t>(i);
      m.short_by_len_[1][static_cast<uint64_t>(b)] = i;
    }
    m.max_short_len_ = 1;
    m.next_id_ = 256;
    return m;
  }

  /// Build from a complete dictionary: token at index i receives id i.
  static LongestPrefixMatcher FromDictionary(const CompactDictionary& dict) {
    LongestPrefixMatcher m;
    size_t n = dict.num_tokens();
    for (size_t i = 0; i < n; ++i) {
      m.InsertInternal(dict.token_ptr(static_cast<Token>(i)), dict.token_len(static_cast<Token>(i)),
                       static_cast<Token>(i));
    }
    m.next_id_ = static_cast<uint32_t>(n);
    return m;
  }

  /// Insert `data` (len bytes) and assign it the next available token id.
  Token Insert(const uint8_t* data, size_t len) {
    Token id = static_cast<Token>(next_id_++);
    InsertInternal(data, len, id);
    return id;
  }

  size_t size() const { return next_id_; }

  /// Longest token whose bytes are a prefix of `data`, with its length.
  std::pair<Token, size_t> FindLongestMatch(const uint8_t* data, size_t data_len) const {
    size_t max_len = std::min(data_len, kMaxTokenSize);
    uint64_t low64 = LoadLeU64(data, data_len, std::min(max_len, kBucketPrefixLen));

    if (max_len > kBucketPrefixLen && !long_map_.empty()) {
      auto it = long_map_.find(low64);
      if (it != long_map_.end()) {
        const uint8_t* suf = data + kBucketPrefixLen;
        size_t suf_len = max_len - kBucketPrefixLen;
        const Bucket& b = it->second;
        std::pair<Token, size_t> hit{0, 0};
        bool found;
        if (b.trie_root < 0) {
          found = SearchLinear(b.entries, LoadLeU64(suf, suf_len, suf_len), suf_len, &hit);
        } else {
          found = SearchTrie(static_cast<uint32_t>(b.trie_root), suf, suf_len, &hit);
        }
        if (found) {
          return {hit.first, kBucketPrefixLen + hit.second};
        }
      }
    }

    size_t short_max = std::min(max_len, static_cast<size_t>(max_short_len_));
    for (size_t len = short_max; len >= 1; --len) {
      uint64_t key = low64 & MaskU64(len);
      const auto& map = short_by_len_[len];
      auto it = map.find(key);
      if (it != map.end()) {
        return {it->second, len};
      }
    }
    // Precondition: every single-byte token is present, so len==1 always hits.
    return {static_cast<Token>(data[0]), 1};
  }

 private:
  // short_by_len_[len] maps the low-`len`-byte packed key to a token, for len 1..8.
  std::unordered_map<uint64_t, Token> short_by_len_[kBucketPrefixLen + 1];
  std::unordered_map<uint64_t, Bucket> long_map_;
  std::vector<TrieNode> pool_;
  uint8_t max_short_len_ = 1;
  uint32_t next_id_ = 0;

  void InsertInternal(const uint8_t* data, size_t len, Token id) {
    if (len <= kBucketPrefixLen) {
      uint64_t key = LoadLeU64(data, len, len);
      short_by_len_[len][key] = id;
      max_short_len_ = std::max<uint8_t>(max_short_len_, static_cast<uint8_t>(len));
      return;
    }
    uint64_t prefix = LoadLeU64(data, len, kBucketPrefixLen);
    size_t slen = len - kBucketPrefixLen;
    uint64_t suffix = LoadLeU64(data + kBucketPrefixLen, slen, slen);
    Bucket& b = long_map_[prefix];
    if (b.trie_root < 0) {
      b.entries.push_back(LongEntry{suffix, static_cast<uint8_t>(slen), id});
      // Keep descending-by-length order so the first linear match is longest.
      std::sort(b.entries.begin(), b.entries.end(),
                [](const LongEntry& a, const LongEntry& c) { return a.slen > c.slen; });
      if (b.entries.size() > kPromoteThreshold) {
        BuildTrie(&b);
      }
    } else {
      uint8_t buf[8];
      std::memcpy(buf, &suffix, 8);
      TrieInsert(static_cast<uint32_t>(b.trie_root), buf, slen, id);
    }
  }

  bool SearchLinear(const std::vector<LongEntry>& entries, uint64_t val, size_t max_slen,
                    std::pair<Token, size_t>* out) const {
    for (const LongEntry& e : entries) {
      size_t elen = e.slen;
      if (elen <= max_slen && MatchingLowBytes(val ^ e.suffix) >= elen) {
        *out = {e.token, elen};
        return true;
      }
    }
    return false;
  }

  bool SearchTrie(uint32_t root, const uint8_t* suf, size_t suf_len,
                  std::pair<Token, size_t>* out) const {
    bool have = false;
    uint32_t cur = root;
    for (size_t pos = 0; pos < suf_len; ++pos) {
      uint32_t child;
      if (!TrieFindChild(cur, suf[pos], &child)) break;
      cur = child;
      if (pool_[cur].token >= 0) {
        *out = {static_cast<Token>(pool_[cur].token), pos + 1};
        have = true;
      }
    }
    return have;
  }

  bool TrieFindChild(uint32_t node, uint8_t byte, uint32_t* out) const {
    for (const auto& kv : pool_[node].children) {
      if (kv.first == byte) {
        *out = kv.second;
        return true;
      }
    }
    return false;
  }

  uint32_t TrieAlloc() {
    uint32_t idx = static_cast<uint32_t>(pool_.size());
    pool_.emplace_back();
    return idx;
  }

  void TrieInsert(uint32_t root, const uint8_t* suf, size_t slen, Token token) {
    uint32_t cur = root;
    for (size_t i = 0; i < slen; ++i) {
      uint32_t child;
      if (TrieFindChild(cur, suf[i], &child)) {
        cur = child;
      } else {
        uint32_t new_idx = TrieAlloc();
        pool_[cur].children.emplace_back(suf[i], new_idx);
        cur = new_idx;
      }
    }
    pool_[cur].token = static_cast<int>(token);
  }

  void BuildTrie(Bucket* b) {
    uint32_t root = TrieAlloc();
    for (const LongEntry& e : b->entries) {
      uint8_t buf[8];
      std::memcpy(buf, &e.suffix, 8);
      TrieInsert(root, buf, e.slen, e.token);
    }
    b->entries.clear();
    b->entries.shrink_to_fit();
    b->trie_root = static_cast<int32_t>(root);
  }
};

// Merge-threshold controller - DEVIATION FROM PAPER (D1; see onpair.h)

class DynamicThresholdController {
 public:
  DynamicThresholdController(size_t capacity, size_t total_bytes, double scan_fraction)
      : capacity_(capacity),
        scan_budget_(static_cast<size_t>(static_cast<double>(total_bytes) * scan_fraction)),
        check_interval_(std::max<size_t>(capacity / 128, 64)),
        next_checkpoint_(check_interval_) {}

  uint8_t get() const { return threshold_; }
  bool budget_exhausted() const { return bytes_scanned_ > scan_budget_; }
  void on_bytes_scanned(size_t n) { bytes_scanned_ += n; }

  void on_entry_created() {
    ++entries_created_;
    if (entries_created_ >= next_checkpoint_) Rebalance();
  }

 private:
  size_t capacity_;
  size_t scan_budget_;
  size_t check_interval_;
  uint8_t threshold_ = 2;
  size_t entries_created_ = 0;
  size_t bytes_scanned_ = 0;
  size_t entries_at_check_ = 0;
  size_t bytes_at_check_ = 0;
  size_t next_checkpoint_;

  void Rebalance() {
    size_t delta_e = entries_created_ - entries_at_check_;
    size_t delta_b = bytes_scanned_ - bytes_at_check_;
    double recent_rate =
        delta_b > 0 ? static_cast<double>(delta_e) / static_cast<double>(delta_b) : 1e9;
    size_t e_rem = capacity_ > entries_created_ ? capacity_ - entries_created_ : 1;
    size_t b_rem = scan_budget_ > bytes_scanned_ ? scan_budget_ - bytes_scanned_ : 1;
    double target_rate = static_cast<double>(e_rem) / static_cast<double>(b_rem);
    double ratio = target_rate > 0.0 ? recent_rate / target_rate : 1e9;

    if (ratio > 2.0 && threshold_ < 255) {
      ++threshold_;
    } else if (ratio < 0.5 && threshold_ > 2) {
      --threshold_;
    }
    entries_at_check_ = entries_created_;
    bytes_at_check_ = bytes_scanned_;
    next_checkpoint_ = entries_created_ + check_interval_;
  }
};

// Seeded PRNG for training-sample shuffle - DEVIATION FROM PAPER (D4)

inline uint64_t SplitMix64(uint64_t* state) {
  uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

/// Partial Fisher-Yates: randomize the first `k` positions of `order`.
void PartialShuffle(std::vector<uint32_t>* order, size_t k, uint64_t seed) {
  size_t n = order->size();
  uint64_t state = seed;
  size_t limit = std::min(k, n);
  for (size_t i = 0; i < limit; ++i) {
    size_t span = n - i;
    size_t j = i + static_cast<size_t>(SplitMix64(&state) % span);
    std::swap((*order)[i], (*order)[j]);
  }
}

// Dictionary finalization

/// Sort tokens bytewise-lexicographically, returning fresh (bytes, offsets).
void SortTokens(const std::vector<uint8_t>& bytes, const std::vector<uint32_t>& offsets,
                std::vector<uint8_t>* out_bytes, std::vector<uint32_t>* out_offsets) {
  size_t n = offsets.size() - 1;
  auto tok_begin = [&](size_t id) { return bytes.data() + offsets[id]; };
  auto tok_len = [&](size_t id) { return offsets[id + 1] - offsets[id]; };

  std::vector<size_t> perm(n);
  std::iota(perm.begin(), perm.end(), 0);
  std::sort(perm.begin(), perm.end(), [&](size_t a, size_t b) {
    size_t la = tok_len(a), lb = tok_len(b);
    int cmp = std::memcmp(tok_begin(a), tok_begin(b), std::min(la, lb));
    if (cmp != 0) return cmp < 0;
    return la < lb;
  });

  out_bytes->clear();
  out_bytes->reserve(bytes.size());
  out_offsets->clear();
  out_offsets->reserve(n + 1);
  out_offsets->push_back(0);
  for (size_t old : perm) {
    out_bytes->insert(out_bytes->end(), tok_begin(old), tok_begin(old) + tok_len(old));
    out_offsets->push_back(static_cast<uint32_t>(out_bytes->size()));
  }
}

/// Append zero padding so the fixed 16-byte over-read of any token is in bounds.
void PadRaw(std::vector<uint8_t>* bytes, const std::vector<uint32_t>& offsets) {
  size_t need = static_cast<size_t>(offsets.back()) + kMaxTokenSize;
  if (bytes->size() < need) bytes->resize(need, 0);
}

// Dictionary construction / training (paper sec 3.2)

struct TrainResult {
  CompactDictionary dict;
  LongestPrefixMatcher lpm;
};

TrainResult Train(const uint8_t* data, const uint32_t* offsets, size_t n, const Config& cfg) {
  size_t dict_capacity = size_t{1} << cfg.max_dict_bits;

  std::vector<uint8_t> dict_bytes;
  dict_bytes.reserve(dict_capacity * kMaxTokenSize);
  std::vector<uint32_t> dict_offsets;
  dict_offsets.reserve(dict_capacity + 1);
  dict_offsets.push_back(0);
  for (uint16_t i = 0; i <= 255; ++i) {
    dict_bytes.push_back(static_cast<uint8_t>(i));
    dict_offsets.push_back(static_cast<uint32_t>(dict_bytes.size()));
  }
  LongestPrefixMatcher lpm = LongestPrefixMatcher::New();

  size_t total_bytes = n == 0 ? 0 : offsets[n];
  size_t capacity = dict_capacity - 256;
  DynamicThresholdController ctrl(capacity, total_bytes, cfg.threshold_fraction);
  uint8_t threshold = ctrl.get();

  std::vector<uint32_t> order(n);
  std::iota(order.begin(), order.end(), 0u);
  // Full Fisher-Yates shuffle of the entire training order (D4 in onpair.h). The
  // dynamic byte budget still stops scanning well before the end, so only a
  // sample is trained on - but drawing that sample from a *full* shuffle avoids
  // skew on sequentially-ordered columns. (The Rust reference crate partial-shuffles
  // only ~0.3n rows and leaves them in the slice's TAIL while the trainer reads from
  // the head, so on ordered data like Customer#000... it trains mostly on
  // low-numbered rows and builds a skewed dictionary. This port already shuffles
  // into the head; a full shuffle matches the reference C++ std::shuffle over all
  // rows and removes any doubt.)
  PartialShuffle(&order, n, cfg.seed);

  std::unordered_map<uint32_t, uint8_t> freq;

  bool full_dictionary = false;
  bool budget_exhausted = false;

  for (uint32_t idx : order) {
    if (full_dictionary || budget_exhausted) break;

    size_t s_start = offsets[idx];
    size_t s_end = offsets[idx + 1];
    if (s_end == s_start) continue;
    const uint8_t* str = data + s_start;
    size_t len = s_end - s_start;

    auto [prev_id, prev_len] = lpm.FindLongestMatch(str, len);
    size_t pos = prev_len;

    ctrl.on_bytes_scanned(prev_len);
    if (ctrl.budget_exhausted()) {
      budget_exhausted = true;
      break;
    }

    while (pos < len) {
      auto [curr_id, curr_len] = lpm.FindLongestMatch(str + pos, len - pos);

      ctrl.on_bytes_scanned(curr_len);
      if (ctrl.budget_exhausted()) {
        budget_exhausted = true;
        break;
      }

      size_t pair_len = prev_len + curr_len;
      if (pair_len <= kMaxTokenSize) {
        uint32_t key = (static_cast<uint32_t>(prev_id) << 16) | static_cast<uint32_t>(curr_id);
        uint8_t& slot = freq[key];
        if (slot < 255) ++slot;  // saturating
        if (slot >= threshold) {
          size_t pair_start = pos - prev_len;
          Token new_id = lpm.Insert(str + pair_start, pair_len);
          dict_bytes.insert(dict_bytes.end(), str + pair_start, str + pos + curr_len);
          dict_offsets.push_back(static_cast<uint32_t>(dict_bytes.size()));

          if (lpm.size() == dict_capacity) {
            full_dictionary = true;
            break;
          }
          ctrl.on_entry_created();
          threshold = ctrl.get();

          freq.erase(key);
          prev_id = new_id;
          prev_len = pair_len;
          pos += curr_len;
          continue;
        }
      }
      prev_id = curr_id;
      prev_len = curr_len;
      pos += curr_len;
    }
  }

  std::vector<uint8_t> sorted_bytes;
  std::vector<uint32_t> sorted_offsets;
  SortTokens(dict_bytes, dict_offsets, &sorted_bytes, &sorted_offsets);
  PadRaw(&sorted_bytes, sorted_offsets);

  CompactDictionary dict;
  dict.bytes = std::move(sorted_bytes);
  dict.offsets = std::move(sorted_offsets);
  dict.RecomputeMaxTokenLen();
  LongestPrefixMatcher final_lpm = LongestPrefixMatcher::FromDictionary(dict);
  return TrainResult{std::move(dict), std::move(final_lpm)};
}

// Parsing: greedy longest-prefix tokenization (paper sec 3.3)

void EncodeStrings(const uint8_t* data, const uint32_t* offsets, size_t n,
                   const LongestPrefixMatcher& lpm, std::vector<uint16_t>* codes,
                   std::vector<uint32_t>* row_offsets) {
  row_offsets->push_back(0);
  for (size_t i = 0; i < n; ++i) {
    size_t s = offsets[i];
    size_t e = offsets[i + 1];
    size_t pos = s;
    while (pos < e) {
      auto [tok, mlen] = lpm.FindLongestMatch(data + pos, e - pos);
      codes->push_back(tok);
      pos += mlen;
    }
    row_offsets->push_back(static_cast<uint32_t>(codes->size()));
  }
}

}  // namespace

// Public API

Column Compress(const uint8_t* bytes, size_t /*bytes_len*/, const uint32_t* offsets,
                size_t num_rows, const Config& cfg) {
  TrainResult tr = Train(bytes, offsets, num_rows, cfg);
  Column col;
  col.dict = std::move(tr.dict);
  col.codes.reserve(num_rows == 0 ? 0 : offsets[num_rows]);
  col.row_offsets.reserve(num_rows + 1);
  EncodeStrings(bytes, offsets, num_rows, tr.lpm, &col.codes, &col.row_offsets);
  return col;
}

size_t DecodedLen(const Column& col) {
  size_t sum = 0;
  for (uint16_t c : col.codes) sum += col.dict.token_len(c);
  return sum;
}

namespace {

// Shared body of DecompressInto, parameterised on the gather-copy width for the
// same reason DecompressPackedFixed is. See CompactDictionary::max_token_len.
template <size_t kCopy>
size_t DecompressIntoFixed(const Column& col, uint8_t* out) {
  const CompactDictionary& dict = col.dict;
  size_t w = 0;
  for (uint16_t code : col.codes) {
    const uint8_t* src = dict.token_ptr(code);
    size_t len = dict.token_len(code);
    std::memcpy(out + w, src, kCopy);  // fixed over-copy, kCopy >= every token
    w += len;
  }
  return w;
}

}  // namespace

size_t DecompressInto(const Column& col, uint8_t* out) {
  const size_t maxlen = col.dict.max_token_len;
  if (maxlen <= 4) return DecompressIntoFixed<4>(col, out);
  if (maxlen <= 8) return DecompressIntoFixed<8>(col, out);
  return DecompressIntoFixed<kMaxTokenSize>(col, out);
}

std::vector<uint8_t> PackValues(const uint32_t* vals, size_t n, size_t bits) {
  std::vector<uint8_t> out((n * bits + 7) / 8 + 4, 0);
  size_t bitpos = 0;
  for (size_t i = 0; i < n; ++i) {
    size_t byte = bitpos >> 3, off = bitpos & 7;
    uint32_t w;
    std::memcpy(&w, out.data() + byte, 4);
    w |= (vals[i] << off);  // vals[i] < 2^bits, bits<=25, off<=7 -> fits in u32
    std::memcpy(out.data() + byte, &w, 4);
    bitpos += bits;
  }
  return out;
}

namespace {

// The gather-copy writes a fixed width per token so the copy length is a compile
// time constant, but that width only has to cover the longest token this
// dictionary actually holds -- not kMaxTokenSize. On corpora whose tokens are
// short the difference dominates decode: c_address averages 1.99 bytes per token,
// so a 16-byte copy moves 8x the bytes it needs to.
//
// Measured, this loop is store-bandwidth-bound. Across five unrelated corpora
// (over-copy factor) x (decode MiB/s) came out constant at ~10.6 GiB/s of store
// traffic, and the corpora with the highest over-copy decode slowest. Narrowing
// the width is therefore worth close to the bytes it saves.
//
// The width is chosen once per stream from the dictionary, so there is no
// per-token branch: a predicate on token length would be nearly free on corpora
// where it always goes one way and expensive on the ones that split (urls sit at
// 41% short, the worst possible mix).
template <size_t kCopy>
size_t DecompressPackedFixed(const CompactDictionary& dict, const uint8_t* packed, size_t ncodes,
                             size_t bits, uint8_t* out) {
  size_t bitpos = 0, w = 0;
  const uint32_t mask = (bits >= 32) ? 0xFFFFFFFFu : ((1u << bits) - 1);
  for (size_t i = 0; i < ncodes; ++i) {
    uint32_t word;
    std::memcpy(&word, packed + (bitpos >> 3), 4);
    uint32_t code = (word >> (bitpos & 7)) & mask;  // unpack the code
    bitpos += bits;
    const uint8_t* src = dict.token_ptr(static_cast<Token>(code));
    size_t len = dict.token_len(static_cast<Token>(code));
    std::memcpy(out + w, src, kCopy);  // fixed over-copy, kCopy >= every token
    w += len;
  }
  return w;
}

// As above but with the code width a compile-time constant, so the mask folds to a
// literal and `bitpos += kBits` strength-reduces. Dispatched once per stream, the
// same way the copy width is.
template <size_t kCopy, size_t kBits>
size_t DecompressPackedFixedBits(const CompactDictionary& dict, const uint8_t* packed,
                                 size_t ncodes, uint8_t* out) {
  constexpr uint32_t kMask = (kBits >= 32) ? 0xFFFFFFFFu : ((uint32_t{1} << kBits) - 1);
  size_t bitpos = 0, w = 0;
  for (size_t i = 0; i < ncodes; ++i) {
    uint32_t word;
    std::memcpy(&word, packed + (bitpos >> 3), 4);
    uint32_t code = (word >> (bitpos & 7)) & kMask;
    bitpos += kBits;
    const uint8_t* src = dict.token_ptr(static_cast<Token>(code));
    size_t len = dict.token_len(static_cast<Token>(code));
    std::memcpy(out + w, src, kCopy);
    w += len;
  }
  return w;
}

// Resolve `bits` to a constant for the widths a trained dictionary can produce
// (kMinDictBits..kMaxDictBits), falling back to the runtime-width loop otherwise so
// no input is rejected.
template <size_t kCopy>
size_t DecompressPackedDispatchBits(const CompactDictionary& dict, const uint8_t* packed,
                                    size_t ncodes, size_t bits, uint8_t* out) {
  switch (bits) {
    case 9: return DecompressPackedFixedBits<kCopy, 9>(dict, packed, ncodes, out);
    case 10: return DecompressPackedFixedBits<kCopy, 10>(dict, packed, ncodes, out);
    case 11: return DecompressPackedFixedBits<kCopy, 11>(dict, packed, ncodes, out);
    case 12: return DecompressPackedFixedBits<kCopy, 12>(dict, packed, ncodes, out);
    case 13: return DecompressPackedFixedBits<kCopy, 13>(dict, packed, ncodes, out);
    case 14: return DecompressPackedFixedBits<kCopy, 14>(dict, packed, ncodes, out);
    case 15: return DecompressPackedFixedBits<kCopy, 15>(dict, packed, ncodes, out);
    case 16: return DecompressPackedFixedBits<kCopy, 16>(dict, packed, ncodes, out);
    default: return DecompressPackedFixed<kCopy>(dict, packed, ncodes, bits, out);
  }
}

}  // namespace

size_t DecompressPacked(const CompactDictionary& dict, const uint8_t* packed, size_t ncodes,
                        size_t bits, uint8_t* out) {
  // Read the width, do not scan for it: an O(tokens) scan here costs 1-3% on
  // dictionaries of 20-60k tokens, which is charged to decode for something a
  // stored format keeps in its header. See CompactDictionary::max_token_len.
  const size_t maxlen = dict.max_token_len;
  // Only widths a single store can carry. A 12-byte copy moves 25% fewer bytes
  // than 16 but needs two stores, and measured that loses 4-6% on every corpus it
  // applied to (c_mktsegment, c_phone, p_container) -- so this is not purely a
  // bandwidth effect and one wide store beats two narrow ones. Narrowing to 8 is
  // worth 25-28% on the corpora that allow it.
  //
  // `out` needs kDecodePadding of slack either way, and dict.bytes is read-padded
  // by kMaxTokenSize, so every width here is in bounds.
  if (maxlen <= 4) return DecompressPackedDispatchBits<4>(dict, packed, ncodes, bits, out);
  if (maxlen <= 8) return DecompressPackedDispatchBits<8>(dict, packed, ncodes, bits, out);
  return DecompressPackedDispatchBits<kMaxTokenSize>(dict, packed, ncodes, bits, out);
}

}  // namespace parquet::onpair
