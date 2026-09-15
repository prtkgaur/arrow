#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Run or summarize the controlled PFOR output-buffer reuse experiment."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

ARMS = ["BM_PforWholeSeqDecode", "BM_PforWholeInterleavedDecode",
        "BM_PforReuseSeqDecode", "BM_PforReuseInterleavedDecode"]
SIZES = {4096: "16 KiB", 102400: "400 KiB", 393216: "1.5 MiB",
         1048576: "4 MiB", 8388608: "32 MiB"}


def summarize(path):
    data = json.loads(path.read_text())["benchmarks"]
    if any(b.get("error_occurred") for b in data):
        raise ValueError("Benchmark reported an error")
    rows = {}
    cvs = {}
    for b in data:
        parts = b.get("run_name", "").split("/")
        if len(parts) != 3 or parts[0] not in ARMS:
            continue
        key = (parts[0], parts[1], int(parts[2]))
        if b.get("aggregate_name") == "median":
            if key in rows:
                raise ValueError(f"Duplicate median: {key}")
            rows[key] = b
        elif b.get("aggregate_name") == "cv":
            cvs[key] = b["cpu_time"]
    datasets = sorted({key[1] for key in rows})
    if not datasets:
        raise ValueError("No output-reuse benchmark medians found")
    expected = {(a, d, n) for a in ARMS for d in datasets for n in SIZES}
    if set(rows) != expected or set(cvs) != expected:
        raise ValueError("Incomplete results: need all four arms and five sizes per column")

    per_column = []
    for ds in datasets:
        for n in SIZES:
            cells = [rows[(a, ds, n)] for a in ARMS]
            if len({b["compressed_bytes"] for b in cells}) != 1:
                raise ValueError(f"Compressed sizes differ: {ds}/{n}")
            if len({b["compression_ratio"] for b in cells}) != 1:
                raise ValueError(f"Compression ratios differ: {ds}/{n}")
            for i, b in enumerate(cells):
                if b["output_buffer_bytes"] != (n * 4 if i < 2 else 4096):
                    raise ValueError(f"Wrong output footprint: {b['run_name']}")
                if b["cpu_time"] <= 0 or b["time_unit"] != "ns":
                    raise ValueError(f"Unexpected timing: {b['run_name']}")
            t = [b["cpu_time"] for b in cells]
            per_column.append({
                "dataset": ds, "values": n,
                "compressed_bytes": cells[0]["compressed_bytes"],
                "whole_interleaved_speedup": t[0] / t[1],
                "reuse_interleaved_speedup": t[2] / t[3],
                "seq_speedup_from_reuse": t[0] / t[2],
                "interleaved_speedup_from_reuse": t[1] / t[3],
                "max_cpu_time_cv": max(cvs[(a, ds, n)] for a in ARMS),
            })

    print(f"Validated {len(rows)} cases across {len(datasets)} columns.")
    print("Ratios use median CPU times; >1 favors interleaving or reuse.")
    print("Logical output   int/seq whole   int/seq reuse   seq reuse gain   int reuse gain")
    metrics = ["whole_interleaved_speedup", "reuse_interleaved_speedup",
               "seq_speedup_from_reuse", "interleaved_speedup_from_reuse"]
    for n, label in SIZES.items():
        subset = [r for r in per_column if r["values"] == n]
        ratios = [math.exp(statistics.mean(math.log(r[k]) for r in subset))
                  for k in metrics]
        print(f"{label:<14}" + "".join(f"{r:>15.3f}x" for r in ratios))
    print(f"Median CPU-time CV: {statistics.median(cvs.values()):.2%}; "
          f"above 5%: {sum(v > .05 for v in cvs.values())}/{len(cvs)}")
    summary = path.with_suffix(".summary.json")
    summary.write_text(json.dumps(per_column, indent=2) + "\n")
    print(f"Per-column ratios and variability: {summary}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--benchmark", type=Path, help="AVX2-built benchmark executable")
    mode.add_argument("--input", type=Path, help="Summarize existing benchmark JSON")
    parser.add_argument("--output", type=Path, default=Path("pfor-output-reuse.json"))
    parser.add_argument("--datasets", default="OS,TpcdsItemSk,SensorDropouts,SortedUnixTime")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--min-time", type=float, default=.2, help="Seconds per repetition")
    parser.add_argument("--cpu", type=int, help="Linux CPU affinity (default: CPU 2 if allowed)")
    args = parser.parse_args()
    if args.input:
        summarize(args.input)
        return
    if args.repetitions < 2 or args.min_time <= 0:
        parser.error("Need at least two repetitions and a positive minimum time")
    binary = args.benchmark.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error(f"Not an executable: {binary}")
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Results already exist; choose another --output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    datasets = args.datasets.split(",")
    expected = {f"{a}/{d}/{n}" for a in ARMS for d in datasets for n in SIZES}
    pattern = "^(" + "|".join(re.escape(a) for a in ARMS) + ")/(" + "|".join(
        re.escape(d) for d in datasets) + ")/(" + "|".join(map(str, SIZES)) + ")$"
    env = dict(os.environ, ARROW_USER_SIMD_LEVEL="AVX2")
    names = subprocess.check_output([str(binary), "--benchmark_list_tests=true",
                                     "--benchmark_filter=" + pattern], env=env, text=True)
    if set(names.splitlines()) != expected:
        parser.error("Registered cases do not match: check the build and dataset names")
    cpu = None
    if hasattr(os, "sched_getaffinity"):
        allowed = os.sched_getaffinity(0)
        cpu = args.cpu if args.cpu is not None else (2 if 2 in allowed else min(allowed))
        if cpu not in allowed:
            parser.error(f"CPU {cpu} is outside the allowed affinity")
        os.sched_setaffinity(0, {cpu})
    elif args.cpu is not None:
        parser.error("--cpu requires Linux CPU-affinity support")
    command = [str(binary), "--benchmark_filter=" + pattern,
               f"--benchmark_repetitions={args.repetitions}",
               f"--benchmark_min_time={args.min_time}s",
               "--benchmark_enable_random_interleaving=true",
               "--benchmark_display_aggregates_only=true",
               "--benchmark_out_format=json", "--benchmark_out=" + str(output)]
    metadata = {"command": command, "cpu": cpu,
                "ARROW_USER_SIMD_LEVEL": "AVX2",
                "executable_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    metadata_path = output.with_suffix(".run.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    log_path = output.with_suffix(".log")
    print(f"Running {len(expected)} cases; console output: {log_path}", flush=True)
    start = time.monotonic()
    with log_path.open("w") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    metadata.update(elapsed_seconds=time.monotonic() - start, exit_code=result.returncode)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if result.returncode:
        raise SystemExit(f"Benchmark failed; see {log_path}")
    summarize(output)


if __name__ == "__main__":
    main()
