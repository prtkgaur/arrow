#!/usr/bin/env python3
"""Render a supplied current-schema CSV; no historical correction factors or conclusions."""
import argparse
import csv
import html
import math
from pathlib import Path
import sys
import statistics

ARMS = ['seq_scal', 'seq_simd', 'intlv', 'fl_unpk', 'fl_tpos', 'pure_st']
POINTS = ['page16k', 'page256k', 'page1m', 'scan4m', 'scan48m', 'batch48m']


def load(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('empty run')
    seen = set()
    for r in rows:
        key = r['dataset'], r['point']
        if key in seen:
            raise ValueError(f'duplicate row: {key}')
        seen.add(key)
        for field in ARMS + ['max_cv', 'control_ratio', 'src_mib', 'dst_mib']:
            value = float(r[field])
            if not math.isfinite(value) or value < 0 or (field in ARMS and value == 0):
                raise ValueError(f'invalid {field}: {key}')
        r['valid'] = (r['valid'] == '1' and float(r['max_cv']) <= .05
                      and 1/1.05 <= float(r['fl_unpk']) / float(r['intlv']) <= 1.05)
    expected = {(d, p) for d, _ in seen for p in POINTS}
    if seen != expected:
        raise ValueError('missing or unexpected footprint(s) for one or more datasets')
    return rows


def verify_raw(rows, path):
    """Check counts, elapsed-time arithmetic and summary statistics against raw samples."""
    with open(path, newline='') as f:
        raw = list(csv.DictReader(f))
    summaries = {(r['dataset'], r['point']): r for r in rows}
    samples, seen, orders = {}, set(), set()
    for r in raw:
        case = r['dataset'], r['point']
        if case not in summaries or r['arm'] not in ARMS:
            raise ValueError('unexpected raw sample')
        summary = summaries[case]
        rep, order, iterations = int(r['repetition']), int(r['order']), int(r['iterations'])
        key = (*case, r['arm'], rep)
        order_key = (*case, rep, order)
        if key in seen or order_key in orders:
            raise ValueError('duplicate raw sample or order position')
        seen.add(key)
        orders.add(order_key)
        if not 0 <= rep < int(summary['repetitions']) or not 0 <= order < len(ARMS) or iterations <= 0:
            raise ValueError('invalid raw repetition/order/iterations')
        seconds, speed = float(r['seconds']), float(r['gibs'])
        if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(speed) or speed <= 0:
            raise ValueError('invalid raw duration/throughput')
        expected = iterations * int(summary['n']) * 4 / seconds / (1 << 30)
        if not math.isclose(speed, expected, rel_tol=1e-8):
            raise ValueError('raw throughput disagrees with bytes, iterations and duration')
        samples.setdefault((*case, r['arm']), []).append(speed)
    for case, row in summaries.items():
        cvs = []
        for arm in ARMS:
            values = samples.get((*case, arm), [])
            if len(values) != int(row['repetitions']) or len(values) < 3:
                raise ValueError('missing raw repetitions')
            if not math.isclose(statistics.median(values), float(row[arm]), rel_tol=1e-7):
                raise ValueError('summary median disagrees with raw repetitions')
            cvs.append(statistics.stdev(values) / statistics.mean(values))
        if not math.isclose(max(cvs), float(row['max_cv']), rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError('summary CV disagrees with raw repetitions')


def render(rows):
    out = ['<!doctype html><meta charset="utf-8"><title>FOR microbenchmark</title>',
           '<h1>FOR microbenchmark</h1><p>Synthetic generated columns. Median GiB/s of decoded output. '
           'These are standalone kernels, not production Parquet throughput. Ratios include dispatch and framing. '
           'Cache residency depends on the machine. The write-only reference omits packed input reads. '
           'No hardware bandwidth attribution or universal calibration factor is applied.</p>']
    out.append(f'<p>{len({r["dataset"] for r in rows})} columns. '
               f'{sum(not r["valid"] for r in rows)} inconclusive rows (control gap or CV &gt; 5%).</p>')
    columns = ['dataset', 'point', 'src_mib', 'dst_mib'] + ARMS + ['max_cv', 'valid']
    out.append('<table><tr>' + ''.join(f'<th>{x}</th>' for x in columns) + '</tr>')
    for r in rows:
        out.append('<tr>' + ''.join(f'<td>{html.escape(str(r[x]))}</td>' for x in columns) + '</tr>')
    out.append('</table><h2>Geometric means over matched columns</h2><table><tr><th>point</th>'
               '<th>intlv/seq_simd</th><th>fl_tpos/seq_simd</th><th>fl_unpk/intlv</th><th>status</th></tr>')
    for point in POINTS:
        subset = [r for r in rows if r['point'] == point]
        ratios = []
        for a, b in [('intlv', 'seq_simd'), ('fl_tpos', 'seq_simd'), ('fl_unpk', 'intlv')]:
            ratios.append(math.exp(sum(math.log(float(r[a])/float(r[b])) for r in subset)/len(subset)))
        status = 'PASS' if all(r['valid'] for r in subset) else 'INCONCLUSIVE'
        out.append(f'<tr><td>{point}</td>' + ''.join(f'<td>{v:.4f}</td>' for v in ratios) + f'<td>{status}</td></tr>')
    return '\n'.join(out) + '</table>\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('csv_path')
    ap.add_argument('html_path')
    args = ap.parse_args()
    try:
        rows = load(args.csv_path)
        verify_raw(rows, args.csv_path + '.raw.csv')
        Path(args.html_path).write_text(render(rows))
    except (OSError, ValueError, KeyError) as e:
        sys.exit(f'invalid benchmark results: {e}')
    return 0 if all(r['valid'] for r in rows) else 2


if __name__ == '__main__':
    sys.exit(main())
