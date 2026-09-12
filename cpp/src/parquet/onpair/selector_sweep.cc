// The decode-aware selector, measured against the bytes-only one it replaces.
//
// Trains every rung on every corpus once, then runs SelectBudget over the same nine
// candidates at a range of caps and reports what the chosen rung actually does --
// measured decode, not predicted. The two columns to read together are median ratio
// and worst-case decode regression: the cap trades one for the other, and the point
// of the table is that the exchange is very lopsided in the cap's favour.
#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "parquet/onpair/bench_common.h"
#include "parquet/onpair/onpair.h"

namespace op = parquet::onpair;
using bench::BitPackedBytes;
using bench::BitWidth;
using bench::IndexBits;

namespace {

size_t StoredBytes(const op::Column& col, const bench::Corpus& c) {
  size_t db = col.dict.logical_bytes();
  return db + BitPackedBytes(col.dict.offsets.size(), std::max<size_t>(1, BitWidth(db))) +
         BitPackedBytes(col.codes.size(), IndexBits(col.dict.num_tokens())) + c.len_array_bytes();
}

struct Rung {
  op::BudgetCandidate cand;
  double ratio = 0;
  double dec_sec = 0;
  size_t dict_bytes = 0;   // logical token bytes
  size_t len_bytes = 0;    // the caller's row-length side array, constant per corpus
  size_t raw_bytes = 0;
  uint8_t max_tok_len = 0;
};

// The two dictionary accountings under comparison. Both charge the same token blob,
// the same packed code stream and the same caller-side row lengths; they differ only
// in how the token boundaries are stored.
//
// Offsets: num_tokens + 1 offsets bit-packed at the width the blob size needs. This is
// what every published ratio in LEARNINGS.md charged.
//
// Lengths: num_tokens lengths at 4 bits each, since a token length is 1..16 by
// invariant, with the offsets recovered by prefix sum when the dictionary is adopted.
// This is what ProposedStringEncodingSpec.md stores.
//
// Neither charges section headers or the streams' trailing pad. Those are equal on
// every rung and identical between the two accountings, so they cannot change a pick
// and would only dilute the comparison.
size_t StoredWithOffsets(const Rung& r) {
  return r.dict_bytes +
         BitPackedBytes(r.cand.num_tokens + 1, std::max<size_t>(1, BitWidth(r.dict_bytes))) +
         BitPackedBytes(r.cand.num_codes, r.cand.code_width) + r.len_bytes;
}

size_t StoredWithLengths(const Rung& r) {
  return r.dict_bytes + BitPackedBytes(r.cand.num_tokens, 4) +
         BitPackedBytes(r.cand.num_codes, r.cand.code_width) + r.len_bytes;
}

// One rung: trained at `b`, round-trip verified, and timed.
Rung TrainRung(const bench::Corpus& c, uint8_t b, bool prune) {
  {
    op::Config cfg;
    cfg.max_dict_bits = b;
    cfg.threshold_fraction = bench::ThresholdFor(c.name);
    cfg.seed = 42;
    cfg.prune_absent_literals = prune;
    op::Column col = op::Compress(c.bytes.data(), c.raw_bytes(), c.offsets.data(), c.n_rows(), cfg);
    const size_t width = IndexBits(col.dict.num_tokens());
    Rung r;
    r.cand.budget = b;
    r.cand.code_width = static_cast<uint8_t>(width);
    r.cand.num_tokens = static_cast<uint32_t>(col.dict.num_tokens());
    r.cand.num_codes = col.codes.size();
    r.cand.stored_bytes = StoredBytes(col, c);
    r.ratio = static_cast<double>(c.raw_bytes()) / static_cast<double>(r.cand.stored_bytes);
    r.dict_bytes = col.dict.logical_bytes();
    r.len_bytes = c.len_array_bytes();
    r.raw_bytes = c.raw_bytes();
    for (size_t t = 0; t + 1 < col.dict.offsets.size(); ++t) {
      const size_t len = col.dict.offsets[t + 1] - col.dict.offsets[t];
      if (len > r.max_tok_len) r.max_tok_len = static_cast<uint8_t>(len);
    }

    std::vector<uint32_t> cw(col.codes.begin(), col.codes.end());
    std::vector<uint8_t> packed = op::PackValues(cw.data(), cw.size(), width);
    std::vector<uint8_t> buf(op::DecodedLen(col) + c.raw_bytes() + op::kDecodePadding, 0);
    std::vector<double> secs;
    for (int it = 0; it < bench::kDecodeIters; ++it) {
      auto t0 = bench::Clock::now();
      size_t w = op::DecompressPacked(col.dict, packed.data(), col.codes.size(), width, buf.data());
      secs.push_back(std::chrono::duration<double>(bench::Clock::now() - t0).count());
      asm volatile("" ::"r"(w) : "memory");
      if (w != c.raw_bytes() || std::memcmp(buf.data(), c.bytes.data(), c.raw_bytes()) != 0) {
        std::fprintf(stderr, "ROUNDTRIP MISMATCH %s b%u\n", c.name.c_str(), b);
        std::abort();
      }
    }
    r.dec_sec = bench::Median(std::move(secs));
    return r;
  }
}

// The nine-rung ladder the selector chooses from: occurring literals only, which is
// what makes budget 8 a usable rung at all.
std::vector<Rung> TrainLadder(const bench::Corpus& c) {
  std::vector<Rung> out;
  for (uint8_t b = 8; b <= 16; ++b) out.push_back(TrainRung(c, b, /*prune=*/true));
  return out;
}

// The published OnPair-auto baseline: budgets 9..16 with all 256 bytes resident,
// chosen on stored bytes alone. Trained in this process so its decode figures are
// comparable with the ladder's -- across binaries they are not.
Rung TrainAutoA(const bench::Corpus& c) {
  Rung best;
  best.cand.stored_bytes = UINT64_MAX;
  for (uint8_t b = 9; b <= 16; ++b) {
    Rung r = TrainRung(c, b, /*prune=*/false);
    if (r.cand.stored_bytes < best.cand.stored_bytes) best = r;
  }
  return best;
}

// The published OnPair-auto medians average the two middle values on an even
// sample; bench::Median takes the upper one. Match the published convention here so
// this table's numbers sit beside 4.276x and 4.441x rather than near them.
double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 == 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
double Quantile(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, static_cast<size_t>(q * v.size()))];
}

// --- The wired path ---------------------------------------------------------
//
// Everything above rebuilds the ladder by hand, which is what let the policy be
// measured before it existed. CompressAuto is the same decision inside the encoder.
// Running both and comparing the chosen budget is the check that the library does
// what the table claims: nothing here reads the numbers above, so a divergence
// shows up as a mismatch count rather than as numbers that quietly differ.

// The corpus's own framing, passed through LadderOptions so the selector ranks rungs
// by the same bytes this driver reports ratios from.
uint64_t StoredBytesHook(const op::CompactDictionary& dict, uint64_t num_codes, void* ctx) {
  const auto& c = *static_cast<const bench::Corpus*>(ctx);
  const size_t db = dict.logical_bytes();
  return db + BitPackedBytes(dict.offsets.size(), std::max<size_t>(1, BitWidth(db))) +
         BitPackedBytes(num_codes, IndexBits(dict.num_tokens())) + c.len_array_bytes();
}

struct AutoRun {
  uint8_t budget = 0;
  size_t raw_bytes = 0;
  uint8_t bytes_only_budget = 0;
  double ratio = 0;
  size_t distinct = 0;
  size_t rungs = 0;
  double encode_s = 0;
};

AutoRun RunAuto(const bench::Corpus& c, double cap) {
  op::LadderOptions opts;
  opts.base.threshold_fraction = bench::ThresholdFor(c.name);
  opts.policy.max_decode_regression = cap;
  opts.stored_bytes = &StoredBytesHook;
  opts.stored_bytes_ctx = const_cast<bench::Corpus*>(&c);

  op::SelectionReport rep;
  op::Column col =
      op::CompressAuto(c.bytes.data(), c.raw_bytes(), c.offsets.data(), c.n_rows(), opts, &rep);

  // The column the encoder hands back has to decode, not merely have been chosen.
  const size_t width = IndexBits(col.dict.num_tokens());
  std::vector<uint32_t> cw(col.codes.begin(), col.codes.end());
  std::vector<uint8_t> packed = op::PackValues(cw.data(), cw.size(), width);
  std::vector<uint8_t> buf(op::DecodedLen(col) + c.raw_bytes() + op::kDecodePadding, 0);
  size_t w = op::DecompressPacked(col.dict, packed.data(), col.codes.size(), width, buf.data());
  if (w != c.raw_bytes() || std::memcmp(buf.data(), c.bytes.data(), c.raw_bytes()) != 0) {
    std::fprintf(stderr, "AUTO ROUNDTRIP MISMATCH %s\n", c.name.c_str());
    std::abort();
  }

  AutoRun a;
  a.budget = rep.candidates[rep.chosen].budget;
  a.bytes_only_budget = rep.candidates[rep.bytes_only].budget;
  a.ratio = static_cast<double>(c.raw_bytes()) /
            static_cast<double>(rep.candidates[rep.chosen].stored_bytes);
  a.distinct = rep.distinct_dictionaries;
  a.rungs = rep.candidates.size();
  a.encode_s = rep.encode_s;
  a.raw_bytes = c.raw_bytes();
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  std::string dir = bench::CorpusDir(argc, argv);
  std::vector<std::vector<Rung>> ladders;
  std::vector<Rung> autoA;
  std::vector<AutoRun> wired;
  std::vector<std::string> names;
  for (const auto& path : bench::CorpusFiles(dir)) {
    bench::Corpus c = bench::ReadCorpus(path);
    if (c.offsets.size() < 2) continue;
    names.push_back(c.name);
    ladders.push_back(TrainLadder(c));
    autoA.push_back(TrainAutoA(c));
    wired.push_back(RunAuto(c, 0.30));
    std::fprintf(stderr, "trained %s\n", c.name.c_str());
  }

  const double kInf = std::numeric_limits<double>::infinity();
  const double caps[] = {0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, kInf};

  std::printf("\n%zu corpora, nine rungs each. Decode is measured; the cap is applied to the\n",
              ladders.size());
  std::printf("prediction, so this is the selector's real behaviour and not its own model.\n\n");
  std::printf("%6s %10s %10s %8s %8s %8s %7s %7s\n", "cap", "med ratio", "med dec", "dec/best",
              "p90", "worst", ">5%", ">20%");
  {
    std::vector<double> ar, am;
    for (size_t i = 0; i < autoA.size(); ++i) {
      ar.push_back(autoA[i].ratio);
      am.push_back(bench::Mib(static_cast<size_t>(autoA[i].ratio * autoA[i].cand.stored_bytes)) /
                   autoA[i].dec_sec);
    }
    std::printf("%6s %9.3fx %10.0f %8s %8s %8s %7s %7s\n", "autoA", Median(ar), Median(am), "-",
                "-", "-", "-", "-");
  }
  for (double cap : caps) {
    op::SelectionPolicy policy;
    policy.max_decode_regression = cap;
    std::vector<double> ratios, mibs, regress;
    int over5 = 0, over20 = 0;
    for (size_t i = 0; i < ladders.size(); ++i) {
      std::vector<op::BudgetCandidate> cands;
      for (const Rung& r : ladders[i]) cands.push_back(r.cand);
      size_t k = op::SelectBudget(cands.data(), cands.size(), policy);
      double fastest = ladders[i][0].dec_sec;
      for (const Rung& r : ladders[i]) fastest = std::min(fastest, r.dec_sec);
      const Rung& sel = ladders[i][k];
      ratios.push_back(sel.ratio);
      // Charge decode against the raw bytes produced, the rate a consumer sees.
      mibs.push_back(bench::Mib(static_cast<size_t>(sel.ratio * sel.cand.stored_bytes)) /
                     sel.dec_sec);
      double reg = sel.dec_sec / fastest;
      regress.push_back(reg);
      if (reg > 1.05) ++over5;
      if (reg > 1.20) ++over20;
    }
    char tag[16];
    if (std::isinf(cap)) {
      std::snprintf(tag, sizeof(tag), "inf");
    } else {
      std::snprintf(tag, sizeof(tag), "%.2f", cap);
    }
    std::printf("%6s %9.3fx %10.0f %7.2fx %7.2fx %7.2fx %7d %7d\n", tag, Median(ratios),
                Median(mibs), Median(regress), Quantile(regress, 0.9), Quantile(regress, 1.0),
                over5, over20);
  }

  // Per column against both the published baseline and the bytes-only ladder, so a
  // median that hides the tail cannot be the whole argument.
  std::printf("\nper column, ratio and decode MiB/s. dominates = beats autoA on both axes.\n");
  std::printf("%-30s %-17s %-17s %-17s %s\n", "corpus", "autoA (published)", "cap 0.05",
              "cap 0.30", "bytes-only");
  op::SelectionPolicy p05, p30, none;
  p05.max_decode_regression = 0.05;
  p30.max_decode_regression = 0.30;
  none.max_decode_regression = kInf;
  int dom05 = 0, dom30 = 0, worse05 = 0, worse30 = 0;
  for (size_t i = 0; i < ladders.size(); ++i) {
    std::vector<op::BudgetCandidate> cands;
    for (const Rung& r : ladders[i]) cands.push_back(r.cand);
    const Rung& a = ladders[i][op::SelectBudget(cands.data(), cands.size(), p05)];
    const Rung& b = ladders[i][op::SelectBudget(cands.data(), cands.size(), p30)];
    const Rung& z = ladders[i][op::SelectBudget(cands.data(), cands.size(), none)];
    const Rung& base = autoA[i];
    auto mibs = [](const Rung& r) {
      return bench::Mib(static_cast<size_t>(r.ratio * r.cand.stored_bytes)) / r.dec_sec;
    };
    if (a.ratio >= base.ratio && mibs(a) >= mibs(base)) ++dom05;
    if (b.ratio >= base.ratio && mibs(b) >= mibs(base)) ++dom30;
    if (a.ratio < base.ratio && mibs(a) < mibs(base)) ++worse05;
    if (b.ratio < base.ratio && mibs(b) < mibs(base)) ++worse30;
    std::printf("%-30s b%-2u %5.3fx %6.0f  b%-2u %5.3fx %6.0f  b%-2u %5.3fx %6.0f  b%-2u %5.3fx %6.0f\n",
                names[i].c_str(), base.cand.budget, base.ratio, mibs(base), a.cand.budget, a.ratio,
                mibs(a), b.cand.budget, b.ratio, mibs(b), z.cand.budget, z.ratio, mibs(z));
  }
  std::printf("\nvs autoA: cap 0.05 dominates %d/30, worse on both %d/30;"
              "  cap 0.30 dominates %d/30, worse on both %d/30\n",
              dom05, worse05, dom30, worse30);

  // CompressAuto at the same cap, against the hand-built ladder above.
  int mismatch = 0, moved = 0;
  size_t distinct = 0, rungs = 0;
  double enc_s = 0, raw = 0;
  std::vector<double> wr;
  for (size_t i = 0; i < wired.size(); ++i) {
    std::vector<op::BudgetCandidate> cands;
    for (const Rung& r : ladders[i]) cands.push_back(r.cand);
    const uint8_t expect = ladders[i][op::SelectBudget(cands.data(), cands.size(), p30)].cand.budget;
    if (wired[i].budget != expect) {
      std::printf("MISMATCH %-28s wired b%u, ladder b%u\n", names[i].c_str(), wired[i].budget,
                  expect);
      ++mismatch;
    }
    if (wired[i].budget != wired[i].bytes_only_budget) ++moved;
    distinct += wired[i].distinct;
    rungs += wired[i].rungs;
    enc_s += wired[i].encode_s;
    raw += static_cast<double>(wired[i].raw_bytes);
    wr.push_back(wired[i].ratio);
  }
  std::printf("\nCompressAuto at cap 0.30: median ratio %.3fx, %d/%zu budgets differ from the\n",
              Median(wr), mismatch, wired.size());
  std::printf("hand-built ladder's pick, decode weighing moved the choice on %d/%zu columns.\n",
              moved, wired.size());
  std::printf("Ladder dedup trained %zu dictionaries for %zu rungs (%.0f%% of rungs repeated an\n",
              distinct, rungs, 100.0 * (1.0 - static_cast<double>(distinct) / rungs));
  std::printf("already-trained dictionary). Whole-ladder encode: %.1f MiB at %.1f MiB/s.\n",
              bench::Mib(static_cast<size_t>(raw)), bench::Mib(static_cast<size_t>(raw)) / enc_s);
  // --- Re-measure: 4-bit token lengths instead of a packed offset array -----------
  //
  // Same trained rungs, scored two ways, and selected twice: an accounting change can
  // move a pick, because it does not shift every rung equally. A wide dictionary pays
  // more per token under offsets (19 bits on a large blob against 4), so the cheaper
  // accounting relieves the wide end of the ladder more than the narrow end.
  std::printf("\n--- 4-bit token lengths vs a packed offset array ---\n\n");
  uint8_t max_tok = 0;
  for (const auto& L : ladders)
    for (const Rung& r : L) max_tok = std::max(max_tok, r.max_tok_len);
  std::printf("longest token over all %zu rungs: %u bytes -- 4 bits holds len-1 for %s\n\n",
              rungs, max_tok, max_tok <= 16 ? "every one" : "NOT all of them");

  std::printf("%-28s %-18s %-18s\n", "corpus", "offsets accounting", "lengths accounting");
  std::vector<double> ro, rl, rl_same, mo, ml;
  int pick_moved = 0;
  double dict_off = 0, dict_len = 0;
  for (size_t i = 0; i < ladders.size(); ++i) {
    std::vector<op::BudgetCandidate> co, cl;
    for (const Rung& r : ladders[i]) {
      op::BudgetCandidate a = r.cand, b = r.cand;
      a.stored_bytes = StoredWithOffsets(r);
      b.stored_bytes = StoredWithLengths(r);
      co.push_back(a);
      cl.push_back(b);
    }
    const size_t ko = op::SelectBudget(co.data(), co.size(), p30);
    const size_t kl = op::SelectBudget(cl.data(), cl.size(), p30);
    const Rung& so = ladders[i][ko];
    const Rung& sl = ladders[i][kl];
    const double raw = static_cast<double>(so.raw_bytes);
    ro.push_back(raw / co[ko].stored_bytes);
    rl_same.push_back(raw / cl[ko].stored_bytes);  // same rung, cheaper dictionary
    rl.push_back(raw / cl[kl].stored_bytes);
    mo.push_back(bench::Mib(so.raw_bytes) / so.dec_sec);
    ml.push_back(bench::Mib(sl.raw_bytes) / sl.dec_sec);
    dict_off += static_cast<double>(BitPackedBytes(
        so.cand.num_tokens + 1, std::max<size_t>(1, BitWidth(so.dict_bytes))));
    dict_len += static_cast<double>(BitPackedBytes(sl.cand.num_tokens, 4));
    if (co[ko].budget != cl[kl].budget) ++pick_moved;
    std::printf("%-28s b%-2u %6.3fx %6.0f   b%-2u %6.3fx %6.0f%s\n", names[i].c_str(),
                co[ko].budget, ro.back(), mo.back(), cl[kl].budget, rl.back(), ml.back(),
                co[ko].budget != cl[kl].budget ? "   <- pick moved" : "");
  }
  std::printf("\nmedian ratio  offsets %.3fx -> lengths %.3fx  (%+.2f%%)\n", Median(ro),
              Median(rl), 100.0 * (Median(rl) / Median(ro) - 1.0));
  std::printf("  of which     same rungs %.3fx (%+.2f%%), the rest is the %d pick(s) that moved\n",
              Median(rl_same), 100.0 * (Median(rl_same) / Median(ro) - 1.0), pick_moved);
  std::printf("median decode offsets %.0f -> lengths %.0f MiB/s  (%+.2f%%)\n", Median(mo),
              Median(ml), 100.0 * (Median(ml) / Median(mo) - 1.0));
  std::printf("token-boundary bytes over the 30 chosen rungs: %.1f MiB -> %.1f MiB (%.1fx less)\n",
              bench::Mib(static_cast<size_t>(dict_off)), bench::Mib(static_cast<size_t>(dict_len)),
              dict_off / std::max(1.0, dict_len));
  return mismatch == 0 ? 0 : 1;
}
