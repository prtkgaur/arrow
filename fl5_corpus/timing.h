#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace bench {
struct Options {
  std::string filter = "all", csv;
  int reps = 7;
  size_t bytes = size_t{1} << 30;
  unsigned seed = 7;
  bool verify = false, exact = false;
  unsigned min_time_ms = 100;
};
inline Options Parse(int argc, char** argv) {
  Options o;
  int positional = 0;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--exact") {
      o.exact = true;
      continue;
    }
    if (arg == "--verify-only") {
      o.verify = true;
      continue;
    }
    if (arg == "--help") {
      printf(
          "usage: %s [dataset-filter|all] [output.csv] [--verify-only] "
          "[--repetitions=N] [--bytes-per-run=N] [--seed=N] [--exact] "
          "[--min-time-ms=N]\n",
          argv[0]);
      std::exit(0);
    }
    if (arg.rfind("--", 0) == 0) {
      const auto eq = arg.find('=');
      const std::string name = arg.substr(0, eq);
      const std::string value = eq == std::string::npos ? "" : arg.substr(eq + 1);
      if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        fprintf(stderr, "invalid option: %s\n", arg.c_str());
        std::exit(1);
      }
      unsigned long long v = 0;
      try {
        v = std::stoull(value);
      } catch (...) {
        std::exit(1);
      }
      if (name == "--repetitions" && v >= 3 && v <= 1000)
        o.reps = static_cast<int>(v);
      else if (name == "--bytes-per-run" && v >= 4096 && v <= (1ull << 40))
        o.bytes = v;
      else if (name == "--min-time-ms" && v >= 1 && v <= 60000)
        o.min_time_ms = static_cast<unsigned>(v);
      else if (name == "--seed" && v <= 0xffffffffu)
        o.seed = static_cast<unsigned>(v);
      else {
        fprintf(stderr, "invalid option: %s\n", arg.c_str());
        std::exit(1);
      }
    } else if (positional++ == 0)
      o.filter = arg;
    else if (positional == 2)
      o.csv = arg;
    else {
      fprintf(stderr, "unexpected argument: %s\n", arg.c_str());
      std::exit(1);
    }
  }
  return o;
}
inline FILE* Open(const std::string& path) {
  if (path.empty()) return nullptr;
  FILE* f = fopen(path.c_str(), "w");
  if (!f) {
    perror(path.c_str());
    std::exit(1);
  }
  return f;
}
inline void Close(FILE* f) {
  if (f && (ferror(f) || fclose(f) != 0)) {
    perror("writing benchmark output");
    std::exit(1);
  }
}
// Prevent repeated stores from being removed or moved out of a timed iteration.
// Both supported build targets use GCC/Clang; this is a compiler, not CPU, fence.
inline void Clobber() { asm volatile("" ::: "memory"); }
struct Stats {
  double median, cv;
};
inline std::vector<Stats> Measure(const std::function<void(size_t)>* arms, int count,
                                  const std::function<void(int)>& prepare, size_t visits,
                                  size_t iterations, size_t output_bytes,
                                  const Options& options, std::mt19937& rng, FILE* raw,
                                  const char* dataset, const char* point,
                                  const char* const* names) {
  std::vector<std::vector<double>> samples(count);
  std::vector<int> order(count);
  std::iota(order.begin(), order.end(), 0);
  std::vector<size_t> counts(count, iterations);
  auto timed = [&](int arm, size_t n) {
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t it = 0; it < n; ++it) {
      arms[arm](it);
      Clobber();
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  };
  // Calibrate outside the retained repetitions. A fixed byte budget alone can
  // give sub-10ms samples on a fast core. Keep a fixed count per arm thereafter.
  std::shuffle(order.begin(), order.end(), rng);
  for (int a : order) {
    prepare(a);
    for (size_t k = 0; k < visits; ++k) {
      arms[a](k);
      Clobber();
    }
    while (true) {
      const double secs = timed(a, counts[a]);
      if (secs >= options.min_time_ms / 1000.0) break;
      counts[a] *= 2;
    }
  }
  for (int rep = 0; rep < options.reps; ++rep) {
    std::shuffle(order.begin(), order.end(), rng);
    for (int position = 0; position < count; ++position) {
      const int a = order[position];
      prepare(a);
      for (size_t k = 0; k < visits; ++k) {
        arms[a](k);
        Clobber();
      }
      const double secs = timed(a, counts[a]);
      const double speed = double(counts[a]) * output_bytes / secs / (1ull << 30);
      if (!(secs > 0) || !std::isfinite(speed) || !(speed > 0)) std::exit(1);
      samples[a].push_back(speed);
      if (raw)
        fprintf(raw, "%s,%s,%s,%d,%d,%zu,%.12g,%.12g\n", dataset, point, names[a], rep,
                position, counts[a], secs, speed);
    }
  }
  std::vector<Stats> result;
  for (auto& values : samples) {
    std::sort(values.begin(), values.end());
    const double mean =
        std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double variance = 0;
    for (double x : values) variance += (x - mean) * (x - mean);
    const size_t n = values.size();
    result.push_back({(values[n / 2] + values[(n - 1) / 2]) / 2,
                      std::sqrt(variance / (n - 1)) / mean});
  }
  return result;
}
}  // namespace bench
