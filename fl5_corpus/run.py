#!/usr/bin/env python3
"""Run a pinned standalone corpus and preserve source, binary, library and host provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


def command(args, cwd=None):
    return subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=True).stdout


def sha(path):
    with open(path, 'rb') as f:
        digest = hashlib.sha256()
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
        return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('binary', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--dataset', default='all')
    ap.add_argument('--core', type=int, default=2)
    ap.add_argument('--simd', choices=['AVX2', 'SSE4_2', 'NONE'],
                    default='AVX2' if platform.machine() == 'x86_64' else 'NONE')
    ap.add_argument('--repetitions', type=int, default=7)
    ap.add_argument('--bytes-per-run', type=int, default=1 << 30)
    ap.add_argument('--min-time-ms', type=int, default=100)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--verify-only', action='store_true')
    args = ap.parse_args()
    binary = args.binary.resolve(strict=True)
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    if hasattr(os, 'sched_getaffinity') and args.core not in os.sched_getaffinity(0):
        ap.error('requested CPU is not in this process affinity mask')
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, ARROW_USER_SIMD_LEVEL=args.simd)
    cmd = ['taskset', '-c', str(args.core), str(binary), args.dataset,
           str(out / 'results.csv'), f'--repetitions={args.repetitions}',
           f'--bytes-per-run={args.bytes_per_run}', f'--seed={args.seed}',
           f'--min-time-ms={args.min_time_ms}', '--exact']
    if args.verify_only:
        cmd.append('--verify-only')
    linked = command(['ldd', str(binary)])
    libraries = {}
    for line in linked.splitlines():
        for token in line.split():
            if token.startswith('/') and Path(token).is_file():
                libraries[token] = sha(token)
    commit = command(['git', 'rev-parse', 'HEAD'], root).strip()
    (out / 'source.diff').write_text(command(['git', 'diff', 'HEAD'], root))
    sources = {str(p.relative_to(root)): sha(p) for p in (root / 'fl5_corpus').glob('*')
               if p.suffix in ('.h', '.cpp', '.py', '.sh')}
    (out / 'sources').mkdir()
    for name in sources:
        shutil.copy2(root / name, out / 'sources' / Path(name).name)
    metadata = dict(command=cmd, commit=commit, source_hashes=sources,
                    binary=str(binary), binary_sha256=sha(binary), libraries=libraries,
                    ldd=linked, platform=platform.platform(), simd=args.simd,
                    cpu=command(['lscpu']), affinity=sorted(os.sched_getaffinity(0)),
                    environment={k: env.get(k) for k in ['ARROW_USER_SIMD_LEVEL', 'LD_LIBRARY_PATH', 'LD_PRELOAD']},
                    governor={str(p): p.read_text().strip() for p in Path('/sys/devices/system/cpu').glob('cpu*/cpufreq/scaling_governor')},
                    started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    manifest = binary.parent / 'build-info.json'
    if manifest.exists():
        metadata['build'] = json.loads(manifest.read_text())
        if metadata['build']['binary_hashes'].get(binary.name) != metadata['binary_sha256']:
            ap.error('binary no longer matches its build manifest')
    (out / 'machine.json').write_text(json.dumps(metadata, indent=2) + '\n')
    start = time.monotonic()
    with (out / 'stdout.txt').open('w') as stdout, (out / 'stderr.txt').open('w') as stderr:
        result = subprocess.run(cmd, env=env, stdout=stdout, stderr=stderr)
    metadata.update(returncode=result.returncode, elapsed_seconds=time.monotonic()-start)
    metadata['artifacts'] = {p.name: sha(p) for p in out.iterdir() if p.name != 'machine.json' and p.is_file()}
    (out / 'machine.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'exit={result.returncode}, {metadata["elapsed_seconds"]:.1f}s, artifacts: {out}')
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
