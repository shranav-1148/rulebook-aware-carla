"""Offline tests for first-second CARLA speed/brake comparison. No CARLA API needed."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
import carla_brake_transition_report as m


def fake_run(folder,mode='position',source='same'):
    folder.mkdir(parents=True,exist_ok=True)
    summary={'status':'completed','collision_event_count':0,'samples':4,
             'position_rmse_m':1.5 if mode=='position' else .8,
             'final_position_error_m':1.2,'max_abs_cross_track_error_m':.12,
             'source_sha256':source,'vehicle_blueprint':'vehicle.tesla.model3',
             'carla_client_version':'0.9.16','carla_server_version':'0.9.16',
             'fixed_dt_s':.05,'lookahead_m':4.,'wheelbase_assumed_m':2.7,
             'initial_velocity_set_once':True,'world_map':'Town01',
             'synthetic_obstacle_spawned':False,'controller_type':mode,
             'position_kp_per_second':.6,'max_speed_correction_mps':2.,
             'max_commanded_speed_mps':12.}
    (folder/'summary.json').write_text(json.dumps(summary),encoding='utf-8')
    with (folder/'tracking_samples.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['t_s','ref_speed_mps','actual_speed_mps','control_brake',
                                      'control_throttle','position_error_m'])
        w.writeheader()
        speeds=[6.,6.4,3.5,2.5] if mode=='position' else [6.,6.3,6.1,6.]
        brakes=[0,0,.2,0] if mode=='position' else [0,0,0,0]
        for i in range(4):
            w.writerow(dict(t_s=i*.05,ref_speed_mps=6,actual_speed_mps=speeds[i],
                            control_brake=brakes[i],control_throttle=.3,position_error_m=.2))
    return summary


class ReportTests(unittest.TestCase):
    def test_speed_transient_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'a';fake_run(path)
            _,rows=m.read_run(path)
            met=m.metrics(rows)
            self.assertAlmostEqual(met['min_early_speed_mps'],2.5)
            self.assertEqual(met['early_brake_ticks'],1)
            self.assertEqual(len(met['severe_speed_drop_events']),2)
            self.assertAlmostEqual(met['max_per_tick_speed_drop_mps'],2.9)

    def test_compare_produces_json_and_optional_figure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fake_run(root/'b');fake_run(root/'c','coast')
            result=m.compare(root/'b',root/'c',root/'out')
            self.assertAlmostEqual(result['position_rmse_change_m'],-.7)
            self.assertFalse(result['repeatability_established'])
            self.assertEqual(result['n_runs_per_condition'],1)
            self.assertTrue((root/'out'/'brake_transition_summary.json').is_file())

    def test_rejects_different_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fake_run(root/'b');fake_run(root/'c','coast','different')
            with self.assertRaisesRegex(ValueError,'source_sha256'):
                m.compare(root/'b',root/'c',root/'out')

    def test_rejects_aborted_or_collision_runs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fake_run(root/'b');fake_run(root/'c','coast')
            f=root/'c'/'summary.json'
            data=json.loads(f.read_text());data['status']='aborted';f.write_text(json.dumps(data))
            with self.assertRaises(ValueError):m.read_run(root/'c')
            data['status']='completed';data['collision_event_count']=1;f.write_text(json.dumps(data))
            with self.assertRaises(ValueError):m.read_run(root/'c')

    def test_rejects_mismatched_controller(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=fake_run(root/'b');b=fake_run(root/'c','coast')
            b['controller_type']='position'
            with self.assertRaises(ValueError):m.verify_compatible(a,b)

    def test_rejects_mismatched_time_axis(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fake_run(root/'b');fake_run(root/'c','coast')
            p=root/'c'/'tracking_samples.csv';txt=p.read_text().replace('0.1,6,6.1','0.11,6,6.1')
            p.write_text(txt)
            with self.assertRaisesRegex(ValueError,'Time axes differ'):
                m.compare(root/'b',root/'c',root/'out')

    def test_rejects_invalid_numerical_csv(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fake_run(root)
            path=root/'tracking_samples.csv'
            path.write_text(path.read_text().replace('3.5','nan'))
            with self.assertRaisesRegex(ValueError,'Nonfinite'):
                m.read_run(root)

    def test_rejects_mismatched_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=fake_run(root/'b');b=fake_run(root/'c','coast')
            b['fixed_dt_s']=.1
            with self.assertRaisesRegex(ValueError,'fixed_dt'):
                m.verify_compatible(a,b)

if __name__=='__main__':
    unittest.main()
