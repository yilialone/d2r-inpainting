"""Synthetic unit tests only; never use these values as experimental results."""
import csv
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("paired_uncertainty", Path(__file__).with_name("paired_uncertainty.py"))
pu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pu)


class Tests(unittest.TestCase):
    def test_bootstrap(self):
        self.assertEqual(pu.bootstrap_mean([2, 2, 2], 1000, .95, 7), (2, 2))
        self.assertEqual(pu.bootstrap_mean([1, 3, 5], 1000, .95, 7),
                         pu.bootstrap_mean([1, 3, 5], 1000, .95, 7))
        self.assertEqual(pu.quantile([0, 10], .25), 2.5)

    def test_equal_object_weight_not_image_weight(self):
        target = {('1', 'a'): {'psnr': 2}, ('1', 'b'): {'psnr': 4}, ('2', 'a'): {'psnr': 9}}
        baseline = {k: {'psnr': 0} for k in target}
        groups = {k: k[0] for k in target}
        rows = pu.object_pairs(target, baseline, groups, ['psnr'])
        self.assertEqual([r['delta_target_minus_baseline'] for r in rows], [3, 9])
        self.assertEqual(sum(r['delta_target_minus_baseline'] for r in rows)/2, 6)

    def test_reject_mismatch(self):
        with self.assertRaises(ValueError):
            pu.require_same_keys({('1', 'a'): 1}, {('2', 'a'): 1}, 'baseline')

    def test_io_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, baseline = root/'target.csv', root/'baseline.csv'
            rows = [dict(image_num=str(i), mask_name='m', psnr_mask=20+i,
                         ssim_mask=.7+i*.01, lpips_mask=.1) for i in range(3)]
            pu.write_csv(target, rows)
            pu.write_csv(baseline, [dict(r, psnr_mask=r['psnr_mask']-2,
                                        ssim_mask=r['ssim_mask']-.02, lpips_mask=.2) for r in reversed(rows)])
            mapping = root/'objects.csv'
            with contextlib.redirect_stdout(io.StringIO()):
                pu.main(['--target', str(target), '--make-object-template', str(mapping)])
            with self.assertRaises(ValueError):
                pu.main(['--target', str(target), '--baseline', 'LaMa='+str(baseline),
                         '--object-map', str(mapping), '--output-dir', str(root/'bad')])
            args = ['--target', str(target), '--baseline', 'LaMa='+str(baseline),
                    '--assume-independent-images', '--resamples', '1000', '--output-dir', str(root/'out')]
            with contextlib.redirect_stdout(io.StringIO()):
                pu.main(args)
            with (root/'out'/'paired_summary.csv').open(encoding='utf-8-sig') as f:
                summary = list(csv.DictReader(f))
            lpips = next(r for r in summary if r['metric']=='lpips_mask')
            self.assertAlmostEqual(float(lpips['delta_target_minus_baseline']), -.1)
            self.assertAlmostEqual(float(lpips['improvement_positive_is_better']), .1)
            with (root/'out'/'paired_image_metrics.csv').open(encoding='utf-8-sig') as f:
                self.assertEqual(len(list(csv.DictReader(f))), 9)
            meta = json.loads((root/'out'/'analysis_metadata.json').read_text(encoding='utf-8'))
            self.assertIn('no training-seed', meta['scope'])
            filled = root/'filled_objects.csv'
            pu.write_csv(filled, [dict(image_num=str(i), mask_name='m', object_id='A' if i<2 else 'B') for i in range(3)])
            with contextlib.redirect_stdout(io.StringIO()):
                pu.main(['--target', str(target), '--baseline', 'LaMa='+str(baseline),
                         '--object-map', str(filled), '--resamples', '1000', '--output-dir', str(root/'grouped')])
            with (root/'grouped'/'paired_summary.csv').open(encoding='utf-8-sig') as f:
                self.assertEqual(next(csv.DictReader(f))['n_objects'], '2')
            with self.assertRaises(FileExistsError):
                pu.main(args)

    def test_reject_nonfinite_duplicate_and_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, rows in [('nan', [dict(image_num='1', mask_name='m', psnr='nan')]),
                               ('duplicate', [dict(image_num='1', mask_name='m', psnr=1)]*2),
                               ('empty', [dict(image_num='', mask_name='m', psnr=1)])]:
                path=Path(tmp)/(name+'.csv')
                pu.write_csv(path, rows)
                with self.assertRaises(ValueError):
                    pu.index_rows(path, ['image_num', 'mask_name'], ['psnr'])


if __name__ == '__main__':
    unittest.main()
