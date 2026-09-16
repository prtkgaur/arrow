"""Failure-path regression checks for benchmark reports (no timing assertions)."""
import copy
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


HERE = Path(__file__).resolve().parent
layout = module('layout', HERE.parent / 'cpp/src/parquet/pfor_layout_tables.py')
corpus = module('corpus', HERE / 'gen_tables.py')


class LayoutReportTest(unittest.TestCase):
    def setUp(self):
        self.data = {'benchmarks': []}
        for arm in layout.ARMS:
            for n in layout.SIZES:
                for stat, value in [('median', 1e9), ('cv', .01)]:
                    self.data['benchmarks'].append(dict(run_name=f'{arm}/OS/{n}',
                        build_simd_level='AVX2', aggregate_name=stat, bytes_per_second=value))

    def test_complete(self):
        values, cvs = layout.read_records(self.data)
        self.assertEqual(len(values), 25)
        self.assertEqual(len(cvs), 25)

    def test_missing_individual_cell(self):
        self.data['benchmarks'].pop()
        with self.assertRaisesRegex(ValueError, 'missing'):
            layout.read_records(self.data)

    def test_duplicate(self):
        self.data['benchmarks'].append(copy.copy(self.data['benchmarks'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            layout.read_records(self.data)

    def test_nonfinite_and_error(self):
        for value in [0, -1, float('nan'), float('inf')]:
            with self.subTest(value=value):
                data = copy.deepcopy(self.data)
                data['benchmarks'][0]['bytes_per_second'] = value
                with self.assertRaises(ValueError):
                    layout.read_records(data)
        self.data['benchmarks'][0]['error_occurred'] = True
        with self.assertRaisesRegex(ValueError, 'failed'):
            layout.read_records(self.data)

    def test_missing_arm_in_second_build(self):
        second = copy.deepcopy(self.data['benchmarks'])
        for r in second:
            r['build_simd_level'] = 'SSE4_2'
        second = [r for r in second if 'PlainInterleaved' not in r['run_name']]
        self.data['benchmarks'].extend(second)
        with self.assertRaisesRegex(ValueError, 'missing'):
            layout.read_records(self.data)


class CorpusReportTest(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(dataset='OS', point=p, src_mib=.1, dst_mib=1,
                          max_cv=.01, control_ratio=1, valid=1,
                          **{a: 10 for a in corpus.ARMS}) for p in corpus.POINTS]

    def load(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.csv'
            with path.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            return corpus.load(path)

    def test_complete_and_no_predetermined_conclusion(self):
        rows = self.load(self.rows)
        report = corpus.render(rows)
        self.assertIn('1 columns', report)
        self.assertNotIn('DRAM delivers', report)
        self.assertNotIn('real columns', report)

    def test_missing_and_duplicate(self):
        for rows in [self.rows[:-1], self.rows + [self.rows[0]]]:
            with self.assertRaises(ValueError):
                self.load(rows)

    def test_nonfinite(self):
        self.rows[0]['seq_simd'] = float('nan')
        with self.assertRaises(ValueError):
            self.load(self.rows)

    def test_false_pass_is_recomputed(self):
        self.rows[0]['fl_unpk'] = 20
        rows = self.load(self.rows)
        self.assertFalse(rows[0]['valid'])
        self.assertIn('INCONCLUSIVE', corpus.render(rows))

    def test_raw_arithmetic_and_missing_repetitions(self):
        rows = self.load(self.rows)
        for r in rows:
            r.update(n=4096, repetitions=3, max_cv=0)
        raw = [dict(dataset=r['dataset'], point=r['point'], arm=arm,
                    repetition=rep, order=i, iterations=100,
                    seconds=100*4096*4/(10*(1<<30)), gibs=10)
               for r in rows for rep in range(3) for i, arm in enumerate(corpus.ARMS)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.csv'
            def write():
                with path.open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=raw[0].keys())
                    writer.writeheader()
                    writer.writerows(raw)
            write()
            corpus.verify_raw(rows, path)
            raw[0]['iterations'] = 1
            write()
            with self.assertRaisesRegex(ValueError, 'throughput disagrees'):
                corpus.verify_raw(rows, path)
            raw.pop(0)
            write()
            with self.assertRaisesRegex(ValueError, 'missing raw'):
                corpus.verify_raw(rows, path)

    def test_escape_generated_labels(self):
        for row in self.rows:
            row['dataset'] = '<script>'
        self.assertIn('&lt;script&gt;', corpus.render(self.load(self.rows)))


if __name__ == '__main__':
    unittest.main()
