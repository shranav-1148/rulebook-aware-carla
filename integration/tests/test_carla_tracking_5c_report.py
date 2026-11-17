"""Offline Milestone 5C report tests; no CARLA connection."""
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

import carla_tracking_5c_report as r


def make_run(root, profile, *, enhanced, rmse, source='same', run_name=None,
             status='completed', count=5, with_derived=False):
    path=root/profile
    if run_name is not None:
        path=path/run_name
    path.mkdir(parents=True,exist_ok=True)
    summary={'status':status,'collision_event_count':0,'source_sha256':source,
             'fixed_dt_s':0.05,'vehicle_blueprint':'vehicle.tesla.model3',
             'world_map':'Town01','samples':count,'duration_s':0.2,
             'lookahead_m':4.,'initial_velocity_set_once':True,
             'position_rmse_m':rmse,'max_position_error_m':rmse*2,
             'final_position_error_m':rmse,'max_abs_cross_track_error_m':.1,
             'max_abs_heading_error_deg':1.,
             'controller_type':'position' if enhanced else None}
    if with_derived:
        summary.update(along_track_rmse_m=rmse,speed_tracking_rmse_mps=.2)
    (path/'summary.json').write_text(json.dumps(summary),encoding='utf-8')
    if not with_derived:
        with (path/'tracking_samples.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['along_track_error_m','ref_speed_mps','actual_speed_mps'])
            w.writeheader()
            for _ in range(count):
                w.writerow({'along_track_error_m':rmse,'ref_speed_mps':6.,'actual_speed_mps':5.8})
    return path/'summary.json'


class TestReport(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.base=self.root/'baseline'
        self.new=self.root/'position'
        for p in r.PROFILES:
            make_run(self.base,p,enhanced=False,rmse=2.0)
            make_run(self.new,p,enhanced=True,rmse=1.0,with_derived=True)

    def test_baseline_metrics_recovered_from_csv(self):
        data=r.read_summary(self.base/'corridor_first'/'summary.json')
        self.assertAlmostEqual(data['along_track_rmse_m'],2.)
        self.assertAlmostEqual(data['speed_tracking_rmse_mps'],.2)

    def test_aggregate_metrics_and_no_fake_repeatability(self):
        rows,out=r.aggregate(self.base,self.new)
        self.assertEqual(len(rows),len(r.METRICS)*2)
        stat=out['corridor_first']['metrics']['position_rmse_m']
        self.assertAlmostEqual(stat['improvement_percent_positive_is_better'],50.)
        self.assertFalse(out['corridor_first']['enhanced_repeatability_estimated'])
        self.assertIsNone(stat['enhanced_sample_std'])

    def test_repeatability_std_only_with_multiple_runs(self):
        make_run(self.new,'corridor_first',enhanced=True,rmse=1.2,
                 run_name='run_002',with_derived=True)
        _,out=r.aggregate(self.base,self.new)
        self.assertEqual(out['corridor_first']['enhanced_runs'],2)
        self.assertIsNotNone(out['corridor_first']['metrics']['position_rmse_m']['enhanced_sample_std'])

    def test_source_mismatch_is_rejected(self):
        s=self.new/'corridor_first'/'summary.json'
        payload=json.loads(s.read_text());payload['source_sha256']='other';s.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError,'source_sha256'):
            r.aggregate(self.base,self.new)

    def test_abort_is_rejected(self):
        s=self.new/'safety_first'/'summary.json'
        payload=json.loads(s.read_text());payload['status']='aborted';s.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError,'expected completed'):
            r.aggregate(self.base,self.new)

    def test_collision_is_rejected(self):
        s=self.new/'safety_first'/'summary.json'
        payload=json.loads(s.read_text());payload['collision_event_count']=1;s.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError,'collision events present'):
            r.aggregate(self.base,self.new)

    def test_mismatched_fixed_dt_is_rejected(self):
        s=self.new/'safety_first'/'summary.json'
        payload=json.loads(s.read_text());payload['fixed_dt_s']=.1;s.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError,'fixed_dt_s'):
            r.aggregate(self.base,self.new)

    def test_not_position_mode_is_rejected(self):
        s=self.new/'safety_first'/'summary.json'
        payload=json.loads(s.read_text());payload['controller_type']='legacy';s.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError,'position controller'):
            r.aggregate(self.base,self.new)

    def test_main_writes_report(self):
        out=self.root/'analysis'
        r.main(['--baseline-root',str(self.base),'--enhanced-root',str(self.new),'--out',str(out)])
        self.assertTrue((out/'tracking_comparison.json').is_file())
        with (out/'tracking_comparison.csv').open(newline='') as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),len(r.METRICS)*2)


if __name__=='__main__':
    unittest.main()
