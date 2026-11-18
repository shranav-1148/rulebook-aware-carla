"""M6B-4B reproducible offline/mocked tests, NOT proof of CARLA driving safety."""
# Import dependency: copy supplies types or behavior used by this test module.
import copy
# Import dependency: csv supplies types or behavior used by this test module.
import csv
# Import dependency: json supplies types or behavior used by this test module.
import json
# Import dependency: math supplies types or behavior used by this test module.
import math
# Import dependency: tempfile supplies types or behavior used by this test module.
import tempfile
# Import dependency: unittest supplies types or behavior used by this test module.
import unittest
# Import dependency: pathlib supplies types or behavior used by this test module.
from pathlib import Path
# Import dependency: types supplies types or behavior used by this test module.
from types import SimpleNamespace as NS
# Import dependency: unittest.mock supplies types or behavior used by this test module.
from unittest.mock import patch

# Import dependency: carla_feedback_reference_m6b4b supplies types or behavior used by this test module.
import carla_feedback_reference_m6b4b as ref
# Import dependency: carla_feedback_track_m6b4b supplies types or behavior used by this test module.
import carla_feedback_track_m6b4b as track
# Import dependency: carla_model_m6b4 supplies types or behavior used by this test module.
import carla_model_m6b4 as model
# Import dependency: integration.test_carla_m6b4 supplies types or behavior used by this test module.
from integration.test_carla_m6b4 import (FakeBlueprint,FakeCarla,FakeWorld,save_fake_trial)


# Helper routine: adjust synthetic trial; exposes a focused operation or synthetic fixture while keeping test intent readable.
def adjust_synthetic_trial(sample_path):
    """Physically plausible small synthetic trace, 85 frames, with 2-3 cm perturbation."""
    # Context/exception scope: install the temporary mock or exception guard needed only for this check.
    with sample_path.open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    x0,y0,yaw0=[float(rows[0][k]) for k in ('x_cr_m','y_cr_m','yaw_cr_rad')]
    x,y,yaw=x0,y0,yaw0
    # Repeated scenario: exercise the related states independently and retain each assertion result.
    for j,r in enumerate(rows):
        vv=float(r['forward_speed_mps'])*.68
        r['forward_speed_mps']=str(vv)
        r['total_speed_mps']=str(vv)
        # Conditional branch: distinguish the expected state or exceptional edge before continuing validation.
        if j:
            vprev=float(rows[j-1]['forward_speed_mps'])
            mid=max(0.,.5*(vv+vprev))
            delta=math.radians(-66.625*float(r['command_steer_normalized']))
            da=.05*mid*math.tan(delta)/3.004648862774714
            x+=.05*mid*math.cos(yaw+.5*da)
            y+=.05*mid*math.sin(yaw+.5*da)
            yaw=model.wrap(yaw+da)
        r['x_cr_m'],r['y_cr_m'],r['yaw_cr_rad']=str(x),str(y),str(yaw)
    # Context/exception scope: install the temporary mock or exception guard needed only for this check.
    with sample_path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader();w.writerows(rows)


# Test class TestFeedbackReference: groups related setup, helpers, and assertions behind one fixture lifecycle.
class TestFeedbackReference(unittest.TestCase):
    # Fixture lifecycle: create or release the temporary state used only by this test class.
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
    # Fixture lifecycle: create or release the temporary state used only by this test class.
    def tearDown(self):self.temp.cleanup()
    # Test method: verifies perturbation between two and three cm without changing the behavior under test.
    def test_perturbation_between_two_and_three_cm(self):
        p=self.reference['model_feasibility']
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertGreater(p['final_lateral_difference_from_empirical_m'],.02)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertLess(p['final_lateral_difference_from_empirical_m'],.03)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertLessEqual(p['max_abs_feasible_model_steer'],.08)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(self.reference['count'],85)
    # Test method: verifies replay is byte independent data not shared dict without changing the behavior under test.
    def test_replay_is_byte_independent_data_not_shared_dict(self):
        ref.load_reference(self.reference_path,self.obs,self.pos,model.digest(self.model_path),self.cal)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertNotEqual(self.reference['samples'][50]['reference_lateral_m'],
                            self.reference['samples'][50]['baseline_lateral_m'])
    # Test method: verifies modified reference rejects tampered waypoint without changing the behavior under test.
    def test_modified_reference_rejects_tampered_waypoint(self):
        d=copy.deepcopy(self.reference);d['samples'][45]['reference_lateral_m']+=.2
        self.reference_path.write_text(json.dumps(d))
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(ValueError):ref.load_reference(self.reference_path,self.obs,self.pos,model.digest(self.model_path),self.cal)
    # Test method: verifies invalid source hash rejected without changing the behavior under test.
    def test_invalid_source_hash_rejected(self):
        d=dict(self.obs,sha256='wrong')
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(ValueError):ref.generate_reference(self.pos,self.cal,d,model.digest(self.model_path))
    # Test method: verifies unsupported steering sign rejected without changing the behavior under test.
    def test_unsupported_steering_sign_rejected(self):
        d=dict(self.cal,carla_command_to_commonroad_steering_sign=1)
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(ValueError):ref.generate_reference(self.pos,d,self.obs,model.digest(self.model_path))
    # Test method: verifies overspeed in reference rejected without changing the behavior under test.
    def test_overspeed_in_reference_rejected(self):
        d=copy.deepcopy(self.pos);d['rows'][40]['forward_speed_mps']='5.'
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(ValueError):ref.generate_reference(d,self.cal,self.obs,model.digest(self.model_path))
    # Test method: verifies reference main offline and no overwrite without changing the behavior under test.
    def test_reference_main_offline_and_no_overwrite(self):
        outfile=self.dir/'out.json'
        argv=['--positive-summary',str(self.pp),'--positive-samples',str(self.pc),'--model',str(self.model_path),'--out',str(outfile)]
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with patch('carla_feedback_reference_m6b4b.old.b.read_stable_observation',return_value=self.obs),\
             patch('carla_feedback_reference_m6b4b.old.b.preflight',side_effect=AssertionError('No CARLA access')):
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertEqual(ref.main(argv),0)
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertEqual(ref.main(argv),2)
    # Test method: verifies feedback command within bounds without changing the behavior under test.
    def test_feedback_command_within_bounds(self):
        base=self.pos['rows'][40]
        err={'lateral_error_m':-.03,'heading_error_deg':-.6}
        on=track.select_command(base,err,self.cal,True)
        off=track.select_command(base,err,self.cal,False)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(off[-1],0)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(on[:1],off[:1])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(on[2],off[2])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertNotEqual(on[1],off[1])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertLessEqual(abs(on[1]),.08)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertLessEqual(abs(on[-1]),.020001)
    # Test method: verifies error zero on exact reference without changing the behavior under test.
    def test_error_zero_on_exact_reference(self):
        refrow=self.reference['samples'][0]
        start={k:float(self.pos['rows'][0][k]) for k in ('x_cr_m','y_cr_m','yaw_cr_rad')}
        err=track.point_error(start,start,refrow)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertAlmostEqual(err['position_error_m'],0,places=9)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertAlmostEqual(err['heading_error_deg'],0,places=9)
    # Test method: verifies offline validator does not connect server without changing the behavior under test.
    def test_offline_validator_does_not_connect_server(self):
        argv=['--positive-summary',str(self.pp),'--positive-samples',str(self.pc),
              '--model',str(self.model_path),'--reference',str(self.reference_path)]
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with patch('carla_feedback_track_m6b4b.b.read_stable_observation',return_value=self.obs),\
             patch('carla_feedback_track_m6b4b.b.preflight',side_effect=AssertionError('No CARLA access')):
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertEqual(track.main(argv),0)
    # Test method: verifies drive confirmation required without changing the behavior under test.
    def test_drive_confirmation_required(self):
        # Repeated scenario: exercise the related states independently and retain each assertion result.
        for opt in (['--execute-track','--feedback','on'],['--confirm-drive']):
            # Context/exception scope: install the temporary mock or exception guard needed only for this check.
            with self.assertRaises(SystemExit) as x:track.main(opt)
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertEqual(x.exception.code,2)
    # Test method: verifies feedback mode required without changing the behavior under test.
    def test_feedback_mode_required(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(SystemExit):track.main(['--execute-track','--confirm-drive'])
    # Test method: verifies bad model validation before server without changing the behavior under test.
    def test_bad_model_validation_before_server(self):
        self.model_path.write_text(json.dumps(dict(self.cal,observation_sha256='bad')))
        argv=['--execute-track','--confirm-drive','--feedback','off',
              '--positive-summary',str(self.pp),'--positive-samples',str(self.pc),
              '--model',str(self.model_path),'--reference',str(self.reference_path)]
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with patch('carla_feedback_track_m6b4b.b.read_stable_observation',return_value=self.obs),\
             patch('carla_feedback_track_m6b4b.b.preflight',side_effect=AssertionError('Do not access CARLA')):
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertEqual(track.main(argv),2)


# Test class TestMockCARLATracking: groups related setup, helpers, and assertions behind one fixture lifecycle.
class TestMockCARLATracking(unittest.TestCase):
    """Mocks enforce cleanup and collision abort; separately simulated motion not real CARLA."""
    setUp=TestFeedbackReference.setUp
    tearDown=TestFeedbackReference.tearDown
    # Helper routine: mock run; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def mock_run(self,feedback='on',stationary=False,collision_at=None,fail_restore=False,start_offset=0.):
        world=FakeWorld();world.inject_collision_at=collision_at;world.fail_restore=fail_restore
        baseline=self.pos['rows'];refdata=self.reference
        # Helper routine: sample; exposes a focused operation or synthetic fixture while keeping test intent readable.
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
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with patch('carla_feedback_track_m6b4b.b.preflight',return_value=(FakeCarla,None,world,FakeBlueprint(),wp,{'mock':True})),\
             patch('carla_feedback_track_m6b4b.b._get_tick',side_effect=lambda w,timeout_s=None:w.tick()),\
             patch('carla_feedback_track_m6b4b.b.sample_actor',side_effect=sample):
            code=track.execute(args,self.obs,self.pos,self.cal,refdata)
        result=json.loads((args.out/'summary.json').read_text())
        return code,result,world,args.out
    # Test method: verifies control run completes and zero feedback without changing the behavior under test.
    def test_control_run_completes_and_zero_feedback(self):
        code,r,world,_=self.mock_run('off')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(code,0)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(r['status'],'completed')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(r['tracking_metrics']['nonzero_feedback_samples'],0)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(r['settings_restored'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(r['cleanup_ok'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertGreater(r['tracking_metrics']['rms_position_error_m'],.005)
    # Test method: verifies feedback run has nonzero corrections without changing the behavior under test.
    def test_feedback_run_has_nonzero_corrections(self):
        code,r,world,_=self.mock_run('on')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(code,0)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(r['status'],'completed')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertGreater(r['tracking_metrics']['nonzero_feedback_samples'],5)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertLessEqual(r['tracking_metrics']['max_abs_steer_command'],.08)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(r['cleanup_ok'])
    # Test method: verifies no motion aborts and cleans without changing the behavior under test.
    def test_no_motion_aborts_and_cleans(self):
        code,r,world,_=self.mock_run('on',stationary=True)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(code,2)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(r['status'],'aborted')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertIn('No longitudinal motion',r['abort_reason'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(r['cleanup_ok'])
    # Test method: verifies collision aborts and cleans without changing the behavior under test.
    def test_collision_aborts_and_cleans(self):
        code,r,world,_=self.mock_run('off',collision_at=10)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(code,2)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(r['collision_count'],1)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(r['cleanup_ok'])
    # Test method: verifies settings restore failure cannot pass without changing the behavior under test.
    def test_settings_restore_failure_cannot_pass(self):
        code,r,world,_=self.mock_run('off',fail_restore=True)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(code,2)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(r['status'],'cleanup_or_restore_failed')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(r['settings_restored'])
    # Test method: verifies initial absolute pose offset aborts without changing the behavior under test.
    def test_initial_absolute_pose_offset_aborts(self):
        code,r,world,_=self.mock_run('off',start_offset=.04)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(code,2)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertIn('Absolute start pose',r['abort_reason'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(r['cleanup_ok'])
    # Test method: verifies compare reports rejects same and swapped without changing the behavior under test.
    def test_compare_reports_rejects_same_and_swapped(self):
        _,ron,_,onpath=self.mock_run('on')
        _,roff,_,offpath=self.mock_run('off')
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(ValueError):
            track.compare_reports(onpath/'summary.json',onpath/'summary.json',self.dir/'compare_bad.json')
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(ValueError):
            track.compare_reports(offpath/'summary.json',onpath/'summary.json',self.dir/'compare_bad.json')
    # Test method: verifies compare reports valid but no improvement not claimed without changing the behavior under test.
    def test_compare_reports_valid_but_no_improvement_not_claimed(self):
        self.mock_run('on');self.mock_run('off')
        comp=track.compare_reports(self.dir/'tracking_on/summary.json',
                                   self.dir/'tracking_off/summary.json',self.dir/'compare.json')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(comp['status'],'comparison_complete')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(comp['conclusion'],'feedback_improvement_not_demonstrated')
    # Test method: verifies compare rejects wrong reference hash without changing the behavior under test.
    def test_compare_rejects_wrong_reference_hash(self):
        self.mock_run('on');self.mock_run('off')
        op=self.dir/'tracking_off/summary.json'
        d=json.loads(op.read_text());d['perturbed_reference_sha256']='wrong';op.write_text(json.dumps(d))
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(ValueError):
            track.compare_reports(self.dir/'tracking_on/summary.json',op,self.dir/'compare.json')
    # Test method: verifies comparison rejects manually falsified metrics without changing the behavior under test.
    def test_comparison_rejects_manually_falsified_metrics(self):
        self.mock_run('on');self.mock_run('off')
        on=self.dir/'tracking_on/summary.json'
        doc=json.loads(on.read_text());doc['tracking_metrics']['rms_position_error_m']*=.5
        on.write_text(json.dumps(doc))
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaisesRegex(ValueError,'disagree'):
            track.compare_reports(on,self.dir/'tracking_off/summary.json',self.dir/'compare.json')
    # Test method: verifies comparison rejects modified offline control csv without changing the behavior under test.
    def test_comparison_rejects_modified_offline_control_csv(self):
        self.mock_run('on');self.mock_run('off')
        p=self.dir/'tracking_off/calibration_samples.csv'
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with p.open(newline='',encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
        rows[10]['feedforward_steer_normalized']='0.08'
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with p.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaisesRegex(ValueError,'baseline control schedule'):
            track.compare_reports(self.dir/'tracking_on/summary.json',
                                  self.dir/'tracking_off/summary.json',self.dir/'compare.json')
    # Test method: verifies result overwrite rejected before preflight without changing the behavior under test.
    def test_result_overwrite_rejected_before_preflight(self):
        out=self.dir/'claimed';out.mkdir();(out/'summary.json').write_text('old')
        argv=['--execute-track','--confirm-drive','--feedback','off',
            '--positive-summary',str(self.pp),'--positive-samples',str(self.pc),
            '--model',str(self.model_path),'--reference',str(self.reference_path), '--out',str(out)]
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with patch('carla_feedback_track_m6b4b.b.read_stable_observation',return_value=self.obs),\
             patch('carla_feedback_track_m6b4b.b.preflight',side_effect=AssertionError('no CARLA')):
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertEqual(track.main(argv),2)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual((out/'summary.json').read_text(),'old')

# Conditional branch: distinguish the expected state or exceptional edge before continuing validation.
if __name__=='__main__':unittest.main()
