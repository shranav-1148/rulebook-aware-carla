"""M6B-4B reproducible offline/mocked tests, NOT proof of CARLA driving safety."""
import copy
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import carla_feedback_reference_m6b4b as ref
import carla_feedback_track_m6b4b as track
import carla_model_m6b4 as model
from test_carla_m6b4 import (FakeBlueprint,FakeCarla,FakeWorld,save_fake_trial)


def adjust_synthetic_trial(sample_path):
    """Physically plausible small synthetic trace, 85 frames, with 2-3 cm perturbation."""
    with sample_path.open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    x0,y0,yaw0=[float(rows[0][k]) for k in ('x_cr_m','y_cr_m','yaw_cr_rad')]
    x,y,yaw=x0,y0,yaw0
    for j,r in enumerate(rows):
        vv=float(r['forward_speed_mps'])*.68
        r['forward_speed_mps']=str(vv)
        r['total_speed_mps']=str(vv)
        if j:
            vprev=float(rows[j-1]['forward_speed_mps'])
            mid=max(0.,.5*(vv+vprev))
            delta=math.radians(-66.625*float(r['command_steer_normalized']))
            da=.05*mid*math.tan(delta)/3.004648862774714
            x+=.05*mid*math.cos(yaw+.5*da)
            y+=.05*mid*math.sin(yaw+.5*da)
            yaw=model.wrap(yaw+da)
        r['x_cr_m'],r['y_cr_m'],r['yaw_cr_rad']=str(x),str(y),str(yaw)
    with sample_path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader();w.writerows(rows)


class TestFeedbackReference(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.dir=Path(self.temp.name)
        self.pp,self.pc=save_fake_trial(self.dir,1)
        self.np,self.nc=save_fake_trial(self.dir,-1)
        adjust_synthetic_trial(self.pc)
        adjust_synthetic_trial(self.nc)
        self.pos=model.read_trial(self.pp,self.pc,1)
        self.neg=model.read_trial(self.np,self.nc,-1)
        self.cal=model.estimate_model(self.pos,self.neg)
        self.model_path=self.dir/'model.json'
        self.model_path.write_text(json.dumps(self.cal),encoding='utf-8')
        self.obs={'body_center_cr_xy_m':[float(self.pos['rows'][0]['x_cr_m']),float(self.pos['rows'][0]['y_cr_m'])],
                  'yaw_cr_rad':float(self.pos['rows'][0]['yaw_cr_rad']),
                  'sha256':'test-obs-hash','source_frame':967030}
        self.reference=ref.generate_reference(self.pos,self.cal,self.obs,model.digest(self.model_path))
        self.reference_path=self.dir/'reference.json'
        self.reference_path.write_text(json.dumps(self.reference),encoding='utf-8')
    def tearDown(self):self.temp.cleanup()
    def test_perturbation_between_two_and_three_cm(self):
        p=self.reference['model_feasibility']
        self.assertGreater(p['final_lateral_difference_from_empirical_m'],.02)
        self.assertLess(p['final_lateral_difference_from_empirical_m'],.03)
        self.assertLessEqual(p['max_abs_feasible_model_steer'],.08)
        self.assertEqual(self.reference['count'],85)
    def test_replay_is_byte_independent_data_not_shared_dict(self):
        ref.load_reference(self.reference_path,self.obs,self.pos,model.digest(self.model_path),self.cal)
        self.assertNotEqual(self.reference['samples'][50]['reference_lateral_m'],
                            self.reference['samples'][50]['baseline_lateral_m'])
    def test_modified_reference_rejects_tampered_waypoint(self):
        d=copy.deepcopy(self.reference);d['samples'][45]['reference_lateral_m']+=.2
        self.reference_path.write_text(json.dumps(d))
        with self.assertRaises(ValueError):ref.load_reference(self.reference_path,self.obs,self.pos,model.digest(self.model_path),self.cal)
    def test_invalid_source_hash_rejected(self):
        d=dict(self.obs,sha256='wrong')
        with self.assertRaises(ValueError):ref.generate_reference(self.pos,self.cal,d,model.digest(self.model_path))
    def test_unsupported_steering_sign_rejected(self):
        d=dict(self.cal,carla_command_to_commonroad_steering_sign=1)
        with self.assertRaises(ValueError):ref.generate_reference(self.pos,d,self.obs,model.digest(self.model_path))
    def test_overspeed_in_reference_rejected(self):
        d=copy.deepcopy(self.pos);d['rows'][40]['forward_speed_mps']='5.'
        with self.assertRaises(ValueError):ref.generate_reference(d,self.cal,self.obs,model.digest(self.model_path))
    def test_reference_main_offline_and_no_overwrite(self):
        outfile=self.dir/'out.json'
        argv=['--positive-summary',str(self.pp),'--positive-samples',str(self.pc),'--model',str(self.model_path),'--out',str(outfile)]
        with patch('carla_feedback_reference_m6b4b.old.b.read_stable_observation',return_value=self.obs),\
             patch('carla_feedback_reference_m6b4b.old.b.preflight',side_effect=AssertionError('No CARLA access')):
            self.assertEqual(ref.main(argv),0)
            self.assertEqual(ref.main(argv),2)
    def test_feedback_command_within_bounds(self):
        base=self.pos['rows'][40]
        err={'lateral_error_m':-.03,'heading_error_deg':-.6}
        on=track.select_command(base,err,self.cal,True)
        off=track.select_command(base,err,self.cal,False)
        self.assertEqual(off[-1],0)
        self.assertEqual(on[:1],off[:1])
        self.assertEqual(on[2],off[2])
        self.assertNotEqual(on[1],off[1])
        self.assertLessEqual(abs(on[1]),.08)
        self.assertLessEqual(abs(on[-1]),.020001)
    def test_error_zero_on_exact_reference(self):
        refrow=self.reference['samples'][0]
        start={k:float(self.pos['rows'][0][k]) for k in ('x_cr_m','y_cr_m','yaw_cr_rad')}
        err=track.point_error(start,start,refrow)
        self.assertAlmostEqual(err['position_error_m'],0,places=9)
        self.assertAlmostEqual(err['heading_error_deg'],0,places=9)
    def test_offline_validator_does_not_connect_server(self):
        argv=['--positive-summary',str(self.pp),'--positive-samples',str(self.pc),
              '--model',str(self.model_path),'--reference',str(self.reference_path)]
        with patch('carla_feedback_track_m6b4b.b.read_stable_observation',return_value=self.obs),\
             patch('carla_feedback_track_m6b4b.b.preflight',side_effect=AssertionError('No CARLA access')):
            self.assertEqual(track.main(argv),0)
    def test_drive_confirmation_required(self):
        for opt in (['--execute-track','--feedback','on'],['--confirm-drive']):
            with self.assertRaises(SystemExit) as x:track.main(opt)
            self.assertEqual(x.exception.code,2)
    def test_feedback_mode_required(self):
        with self.assertRaises(SystemExit):track.main(['--execute-track','--confirm-drive'])
    def test_bad_model_validation_before_server(self):
        self.model_path.write_text(json.dumps(dict(self.cal,observation_sha256='bad')))
        argv=['--execute-track','--confirm-drive','--feedback','off',
              '--positive-summary',str(self.pp),'--positive-samples',str(self.pc),
              '--model',str(self.model_path),'--reference',str(self.reference_path)]
        with patch('carla_feedback_track_m6b4b.b.read_stable_observation',return_value=self.obs),\
             patch('carla_feedback_track_m6b4b.b.preflight',side_effect=AssertionError('Do not access CARLA')):
            self.assertEqual(track.main(argv),2)


class TestMockCARLATracking(unittest.TestCase):
    """Mocks enforce cleanup and collision abort; separately simulated motion not real CARLA."""
    setUp=TestFeedbackReference.setUp
    tearDown=TestFeedbackReference.tearDown
    def mock_run(self,feedback='on',stationary=False,collision_at=None,fail_restore=False,start_offset=0.):
        world=FakeWorld();world.inject_collision_at=collision_at;world.fail_restore=fail_restore
        baseline=self.pos['rows'];refdata=self.reference
        def sample(w,a,carla,bb,expected_frame=None):
            source=baseline[0 if stationary else min(w.drive_ticks,84)]
            pose={k:float(source[k]) for k in ('x_cr_m','y_cr_m','yaw_cr_rad')}
            pose['x_cr_m']+=start_offset
            return dict(frame=w.frame,t_sim_s=w.sim_t,**pose,
                        forward_speed_mps=0. if stationary else float(source['forward_speed_mps']),
                        lateral_speed_mps=0.,vertical_speed_mps=0.,
                        total_speed_mps=0. if stationary else float(source['forward_speed_mps']),
                        wheel_front_left_deg=0. if stationary else float(source['wheel_front_left_deg']),
                        wheel_front_right_deg=0. if stationary else float(source['wheel_front_right_deg']))
        wp=NS(location=NS(z=0.))
        args=NS(timeout=10.,model=self.model_path,reference=self.reference_path,
                feedback=feedback,out=self.dir/('tracking_'+feedback))
        with patch('carla_feedback_track_m6b4b.b.preflight',return_value=(FakeCarla,None,world,FakeBlueprint(),wp,{'mock':True})),\
             patch('carla_feedback_track_m6b4b.b._get_tick',side_effect=lambda w,timeout_s=None:w.tick()),\
             patch('carla_feedback_track_m6b4b.b.sample_actor',side_effect=sample):
            code=track.execute(args,self.obs,self.pos,self.cal,refdata)
        result=json.loads((args.out/'summary.json').read_text())
        return code,result,world,args.out
    def test_control_run_completes_and_zero_feedback(self):
        code,r,world,_=self.mock_run('off')
        self.assertEqual(code,0)
        self.assertEqual(r['status'],'completed')
        self.assertEqual(r['tracking_metrics']['nonzero_feedback_samples'],0)
        self.assertTrue(r['settings_restored'])
        self.assertTrue(r['cleanup_ok'])
        self.assertGreater(r['tracking_metrics']['rms_position_error_m'],.005)
    def test_feedback_run_has_nonzero_corrections(self):
        code,r,world,_=self.mock_run('on')
        self.assertEqual(code,0)
        self.assertEqual(r['status'],'completed')
        self.assertGreater(r['tracking_metrics']['nonzero_feedback_samples'],5)
        self.assertLessEqual(r['tracking_metrics']['max_abs_steer_command'],.08)
        self.assertTrue(r['cleanup_ok'])
    def test_no_motion_aborts_and_cleans(self):
        code,r,world,_=self.mock_run('on',stationary=True)
        self.assertEqual(code,2)
        self.assertEqual(r['status'],'aborted')
        self.assertIn('No longitudinal motion',r['abort_reason'])
        self.assertTrue(r['cleanup_ok'])
    def test_collision_aborts_and_cleans(self):
        code,r,world,_=self.mock_run('off',collision_at=10)
        self.assertEqual(code,2)
        self.assertEqual(r['collision_count'],1)
        self.assertTrue(r['cleanup_ok'])
    def test_settings_restore_failure_cannot_pass(self):
        code,r,world,_=self.mock_run('off',fail_restore=True)
        self.assertEqual(code,2)
        self.assertEqual(r['status'],'cleanup_or_restore_failed')
        self.assertFalse(r['settings_restored'])
    def test_initial_absolute_pose_offset_aborts(self):
        code,r,world,_=self.mock_run('off',start_offset=.04)
        self.assertEqual(code,2)
        self.assertIn('Absolute start pose',r['abort_reason'])
        self.assertTrue(r['cleanup_ok'])
    def test_compare_reports_rejects_same_and_swapped(self):
        _,ron,_,onpath=self.mock_run('on')
        _,roff,_,offpath=self.mock_run('off')
        with self.assertRaises(ValueError):
            track.compare_reports(onpath/'summary.json',onpath/'summary.json',self.dir/'compare_bad.json')
        with self.assertRaises(ValueError):
            track.compare_reports(offpath/'summary.json',onpath/'summary.json',self.dir/'compare_bad.json')
    def test_compare_reports_valid_but_no_improvement_not_claimed(self):
        self.mock_run('on');self.mock_run('off')
        comp=track.compare_reports(self.dir/'tracking_on/summary.json',
                                   self.dir/'tracking_off/summary.json',self.dir/'compare.json')
        self.assertEqual(comp['status'],'comparison_complete')
        self.assertEqual(comp['conclusion'],'feedback_improvement_not_demonstrated')
    def test_compare_rejects_wrong_reference_hash(self):
        self.mock_run('on');self.mock_run('off')
        op=self.dir/'tracking_off/summary.json'
        d=json.loads(op.read_text());d['perturbed_reference_sha256']='wrong';op.write_text(json.dumps(d))
        with self.assertRaises(ValueError):
            track.compare_reports(self.dir/'tracking_on/summary.json',op,self.dir/'compare.json')
    def test_comparison_rejects_manually_falsified_metrics(self):
        self.mock_run('on');self.mock_run('off')
        on=self.dir/'tracking_on/summary.json'
        doc=json.loads(on.read_text());doc['tracking_metrics']['rms_position_error_m']*=.5
        on.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError,'disagree'):
            track.compare_reports(on,self.dir/'tracking_off/summary.json',self.dir/'compare.json')
    def test_comparison_rejects_modified_offline_control_csv(self):
        self.mock_run('on');self.mock_run('off')
        p=self.dir/'tracking_off/calibration_samples.csv'
        with p.open(newline='',encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
        rows[10]['feedforward_steer_normalized']='0.08'
        with p.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        with self.assertRaisesRegex(ValueError,'baseline control schedule'):
            track.compare_reports(self.dir/'tracking_on/summary.json',
                                  self.dir/'tracking_off/summary.json',self.dir/'compare.json')
    def test_result_overwrite_rejected_before_preflight(self):
        out=self.dir/'claimed';out.mkdir();(out/'summary.json').write_text('old')
        argv=['--execute-track','--confirm-drive','--feedback','off',
            '--positive-summary',str(self.pp),'--positive-samples',str(self.pc),
            '--model',str(self.model_path),'--reference',str(self.reference_path), '--out',str(out)]
        with patch('carla_feedback_track_m6b4b.b.read_stable_observation',return_value=self.obs),\
             patch('carla_feedback_track_m6b4b.b.preflight',side_effect=AssertionError('no CARLA')):
            self.assertEqual(track.main(argv),2)
        self.assertEqual((out/'summary.json').read_text(),'old')

if __name__=='__main__':unittest.main()
