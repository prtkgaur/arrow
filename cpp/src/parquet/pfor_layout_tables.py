#!/usr/bin/env python3
"""Report production layout and standalone ordering separately; reject incomplete runs."""
import argparse
import json
import math
import re
import sys

ARMS = {
    "BM_PforPlainSeqDecode": "seq",
    "BM_PforPlainInterleavedDecode": "interleaved",
    "BM_InterleavedPforDecode": "grid",
    "BM_InterleavedPforFlOrderRawDecode": "raw",
    "BM_InterleavedPforFlOrderDecode": "restored",
}
SIZES = {4096, 102400, 393216, 1048576, 8388608}
NAME = re.compile(r"^(BM_[A-Za-z0-9]+)/([^/]+)/(\d+)(?:_\w+)?$")


def geomean(values):
    return math.exp(sum(map(math.log, values)) / len(values))


def read_records(data, stat="median"):
    records, cvs = {}, {}
    for b in data.get("benchmarks", []):
        m = NAME.match(b.get("run_name") or b.get("name", ""))
        if not m or m[1] not in ARMS:
            continue
        if b.get("error_occurred"):
            raise ValueError(f"benchmark failed: {b}")
        key = (b.get("build_simd_level", b.get("simd_level", "unknown")), m[2], int(m[3]), ARMS[m[1]])
        aggregate = b.get("aggregate_name")
        if aggregate not in (stat, "cv"):
            continue
        value = b.get("bytes_per_second")
        if value is None or not math.isfinite(value) or value < 0 or (aggregate == stat and value == 0):
            raise ValueError(f"invalid throughput: {key}")
        target = records if aggregate == stat else cvs
        if key in target:
            raise ValueError(f"duplicate {aggregate}: {key}")
        target[key] = value
    if not records:
        raise ValueError("no matching measurements")
    levels = sorted({k[0] for k in records})
    all_datasets = {k[1] for k in records}
    for level in levels:
        for dataset in all_datasets:
            for size in SIZES:
                for arm in ARMS.values():
                    key = level, dataset, size, arm
                    if key not in records or key not in cvs:
                        raise ValueError(f"missing measurement or CV: {key}")
    if {k[2] for k in records} != SIZES:
        raise ValueError("unexpected footprint ladder")
    return records, cvs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_path")
    ap.add_argument("--stat", choices=["median", "mean"], default="median")
    args = ap.parse_args()
    try:
        with open(args.json_path) as f:
            records, cvs = read_records(json.load(f), args.stat)
    except (ValueError, OSError) as e:
        sys.exit(str(e))
    print("Synthetic columns; GiB/s of decoded output. Higher ratios favour the numerator.")
    print("Production interleaved/seq compares delta-disabled PFOR implementations.")
    print("Standalone raw/grid and raw/restored compare ordering only; no production denominator.")
    bad = False
    for level in sorted({k[0] for k in records}):
        datasets = sorted({k[1] for k in records if k[0] == level})
        print(f"\nBuild: {level}; {len(datasets)} columns (not necessarily the full corpus)")
        if level == "AVX512":
            print("WARNING: sequential dispatch remains capped at AVX2; widths are asymmetric.")
        print("column output_KiB prod_interleaved/seq raw/grid raw/restored maxCV status")
        for size in sorted(SIZES):
            ratios = []
            for ds in datasets:
                v = {a: records[level, ds, size, a] for a in ARMS.values()}
                cv = max(cvs[level, ds, size, a] for a in ARMS.values())
                row = (v["interleaved"] / v["seq"], v["raw"] / v["grid"], v["raw"] / v["restored"])
                valid = 1 / 1.05 <= row[1] <= 1.05 and cv <= .05
                bad |= not valid
                ratios.append(row)
                print(f"{ds} {size / 256:g} " + " ".join(f"{r:.4f}" for r in row) + f" {cv:.2%} {'PASS' if valid else 'INCONCLUSIVE'}")
            print(f"GEOMEAN {size / 256:g} " + " ".join(f"{geomean([r[i] for r in ratios]):.4f}" for i in range(3)))
    return 2 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
