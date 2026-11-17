"""M6B-6 offline/controller/mock tests. No real CARLA server accessed."""
import csv
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import carla_speed_fit_m6b6 as fit
import carla_speed_track_m6b6 as tr
import carla_route_reference_m6b5 as ref
import test_carla_m6b5 as base_tests
from test_carla_m6b4 import FakeWorld, FakeCarla, FakeBlueprint


def source_fixture(directory,sign):
    d=Path(directory)
    s=d/('source_positive' if sign==1 else 'source_negative');s.mkdir()
    summary={'status':'completed','mode':'manual_steering','samples':85,'collision_count':0,
        'cleanup_ok':True,'settings_restored':True,'throttle_motion_gate':{'passed':True},
        'steering_pulse_gate':{'passed':True},'steering_diagnostics':{'requested_steer_sign':sign},
        'observation_sha256':'stable-observation-hash',
        'configuration':{'estimated_wheel_geometry':{'wheelbase_from_xy_m':3.004648862774714}}}
    sp=s/'summary.json';sp.write_text(json.dumps(summary))
    path=s/'calibration_samples.csv'
    rows=[];v=0.0
    for j in range(85):
        if j==0: phase,th,br='initial',0.,0.
        elif j<=22:phase,th,br='throttle_straight',.25,0.
        elif j<=36:phase,th,br='steer_pulse',.2,0.
        elif j<=44:phase,th,br='neutralize',0.,0.
        else:phase,th,br='brake',0.,.65
        # Fixture has multiple brake-speed samples and a stopped endpoint.
        if j:
            a=1.0 if j<=22 else (.6 if j<=36 else (-.3 if j<=44 else -2.5))
            v=max(0.,v+a*.05)
        rows.append({'frame':100+j,'t_sim_s':10+j*.05,'t_s':j*.05,
            'forward_speed_mps':v,'phase':phase,'command_throttle':th,'command_brake':br,
            'applied_throttle':th,'applied_brake':br,
            'applied_manual_gear_shift':'True' if j else 'False',
            'applied_gear':1 if j else 0,
            'command_steer_normalized':sign*.08 if 23<=j<=36 else 0.})
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    return sp,path


class SpeedFitTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.d=Path(self.tmp.name)
        self.ps,self.pc=source_fixture(self.d,1)
        self.ns,self.nc=source_fixture(self.d,-1)
    def tearDown(self):self.tmp.cleanup()
    def build(self):
        return fit.response([fit.read_trial(self.ps,self.pc,1),fit.read_trial(self.ns,self.nc,-1)])
    def test_fit_from_independent_sources(self):
        r=self.build()
        self.assertGreater(r['phase_response_mps2']['throttle_straight']['median_acceleration_mps2'],0)
        self.assertLess(r['phase_response_mps2']['brake']['mean_acceleration_mps2'],0)
        self.assertEqual(r['status'],'descriptive_offline_fit')
    def test_no_overwrite_and_validate(self):
        p=self.d/'fit.json'
        args=['--positive-summary',str(self.ps),'--positive-samples',str(self.pc),
            '--negative-summary',str(self.ns),'--negative-samples',str(self.nc),
            '--out',str(p)]
        self.assertEqual(fit.main(args),0)
        self.assertEqual(fit.main(args),2)
        self.assertEqual(fit.validate_fit(p,self.ps,self.pc,self.ns,self.nc)['status'],'descriptive_offline_fit')
    def test_tampered_fit_rejected(self):
        p=self.d/'fit.json';r=self.build();r['observation_sha256']='tampered';p.write_text(json.dumps(r))
        with self.assertRaises(ValueError):fit.validate_fit(p,self.ps,self.pc,self.ns,self.nc)
    def test_negative_sign_rejected(self):
        with self.assertRaises(ValueError):fit.read_trial(self.ps,self.pc,-1)
    def test_duplicate_source_rejected(self):
        with self.assertRaises(ValueError):fit.response([fit.read_trial(self.ps,self.pc,1)]*2)
    def test_nonconsecutive_frames_rejected(self):
        with self.nc.open(newline='',encoding='utf-8') as f:r=list(csv.DictReader(f))
        r[30]['frame']='999'
        with self.nc.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=r[0]);w.writeheader();w.writerows(r)
        with self.assertRaises(ValueError):fit.read_trial(self.ns,self.nc,-1)
    def test_wrong_gear_rejected(self):
        with self.nc.open(newline='',encoding='utf-8') as f:r=list(csv.DictReader(f))
        r[23]['applied_gear']='0'
        with self.nc.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=r[0]);w.writeheader();w.writerows(r)
        with self.assertRaises(ValueError):fit.read_trial(self.ns,self.nc,-1)

class SpeedTrackTest(unittest.TestCase):
    def setUp(self):
        base_tests.TestM6B5.setUp(self)
        self.fit_doc={'status':'descriptive_offline_fit', 'observation_sha256':self.obs['sha256'],
            'limits':{'manual_first_gear_only':True}}
        self.fitp=self.dir/'longfit.json';self.fitp.write_text(json.dumps(self.fit_doc))
        self.refp.write_text(json.dumps(self.reference))
        self.reference['_loaded_from']=str(self.refp)
    def tearDown(self):base_tests.TestM6B5.tearDown(self)
    def cmd(self,j=20,last_v=0.3,on=True):
        t=self.reference['samples'][j]
        last={'forward_speed_mps':last_v,'heading_error_deg':0.,'lateral_error_m':0.}
        return tr.command_for_reference(t,last,self.model,self.fit_doc,on)
    def test_control_off_is_identical_to_6b5(self):
        for j in range(1,85):
            th,steer,br,ff,_,_,_=self.cmd(j,on=False)
            t=self.reference['samples'][j]
            self.assertAlmostEqual(th,t['command_throttle']);self.assertAlmostEqual(br,t['command_brake'])
            self.assertAlmostEqual(steer,t['feedforward_steer_normalized'])
    def test_no_simultaneous_throttle_brake(self):
        for j in range(1,85):
            for v in [0,.5,1.5,1.95]:
                th,steer,br,ff,ct,cb,_=self.cmd(j,v,True)
                self.assertLessEqual(th,.25+1e-9);self.assertLessEqual(br,.65+1e-9)
                self.assertGreaterEqual(th,0);self.assertGreaterEqual(br,0)
                self.assertLessEqual(th*br,1e-10)
                self.assertLessEqual(abs(ct),.10+1e-9)
                self.assertLessEqual(abs(cb),.25+1e-9)
    def test_adjusts_throttle_for_overspeed(self):
        self.assertLess(self.cmd(20,1.6,True)[0],self.cmd(20,1.6,False)[0])
    def test_adjusts_throttle_for_underspeed(self):
        self.assertGreater(self.cmd(30,0.,True)[0],self.cmd(30,0.,False)[0])
    def test_reduces_brake_for_underspeed(self):
        self.assertLess(self.cmd(45,0.,True)[2],self.cmd(45,0.,False)[2])
    def test_coast_small_overspeed_never_brakes(self):
        # Real-world regression: at t=1.90, a ~0.001825 brake command
        # corresponded with a 1.047 m/s speed loss in one CARLA tick.
        t=self.cmd(38,1.6051408169903532,True)
        self.assertEqual(t[0],0.)
        self.assertEqual(t[2],0.)
        self.assertEqual(t[5],0.)
    def test_coast_large_overspeed_never_brakes(self):
        # No confidence in mapping small normalized brake to deceleration.
        t=self.cmd(40,1.95,True)
        self.assertEqual(t[0],0.)
        self.assertEqual(t[2],0.)
        self.assertEqual(t[5],0.)
    def test_coast_underspeed_never_throttles_or_brakes(self):
        t=self.cmd(40,0.5,True)
        self.assertEqual(t[0],0.)
        self.assertEqual(t[2],0.)
        self.assertEqual(t[4],0.)
        self.assertEqual(t[5],0.)
    def test_never_throttles_during_brake(self):
        self.assertEqual(self.cmd(45,0.,True)[0],0.)
    def test_reject_unsafe_preceding_speed(self):
        with self.assertRaises(RuntimeError):self.cmd(20,3.,True)
    def test_drive_requires_double_confirm(self):
        for a in (['--execute-track','--speed-feedback','on'], ['--confirm-drive']):
            with self.assertRaises(SystemExit) as cm:tr.main(a)
            self.assertEqual(cm.exception.code,2)
    def test_speed_metrics_math(self):
        rows=[{'speed_error_mps':.1,'throttle_correction':0.,'brake_correction':.05},
              {'speed_error_mps':-.1,'throttle_correction':.02,'brake_correction':0.}]
        out=tr.speed_metrics(rows)
        self.assertAlmostEqual(out['rms_speed_error_mps'],.1)
        self.assertEqual(out['nonzero_throttle_correction_samples'],1)
        self.assertEqual(out['nonzero_brake_correction_samples'],1)
    def run_mock(self,mode='off',stationary=False,collision_at=None,restore_fail=False,measured_v_offset=0):
        world=FakeWorld();world.inject_collision_at=collision_at;world.fail_restore=restore_fail
        def sample(world,actor,carla,bb,expected_frame=None):
            j=0 if stationary else min(world.drive_ticks,84)
            target=self.reference['samples'][j]
            v=0 if stationary else max(0.,target['reference_speed_mps']+measured_v_offset if j<44 else target['reference_speed_mps'])
            return {'frame':world.frame,'t_sim_s':world.sim_t,
                'x_cr_m':target['x_cr_m'],'y_cr_m':target['y_cr_m'],
                'yaw_cr_rad':target['yaw_cr_rad'],
                'forward_speed_mps':v,'lateral_speed_mps':0.,'vertical_speed_mps':0.,
                'total_speed_mps':v,'wheel_front_left_deg':0.,'wheel_front_right_deg':0.}
        args=NS(timeout=10.,speed_feedback=mode,out=self.dir/('trial_'+mode),
                longitudinal_fit=self.fitp,model=self.mp,reference=self.refp)
        wp=NS(location=NS(z=0.))
        with patch('carla_speed_track_m6b6.b.preflight',return_value=(FakeCarla,None,world,FakeBlueprint(),wp,{'mock':True})),\
             patch('carla_speed_track_m6b6.b._get_tick',side_effect=lambda w,timeout_s=None:w.tick()),\
             patch('carla_speed_track_m6b6.b.sample_actor',side_effect=sample):
            rc=tr.execute(args,self.obs,self.model,self.reference,self.geom,self.fit_doc)
        summary=json.loads((args.out/'summary.json').read_text())
        return rc,summary,args.out,world
    def test_mock_off_completed(self):
        rc,r,p,w=self.run_mock()
        self.assertEqual(rc,0)
        self.assertEqual(r['status'],'completed')
        self.assertEqual(r['speed_metrics']['nonzero_brake_correction_samples'],0)
        self.assertTrue(r['cleanup_ok']);self.assertTrue(r['settings_restored'])
    def test_mock_on_completed(self):
        rc,r,p,w=self.run_mock('on',measured_v_offset=-.15)
        self.assertEqual(rc,0);self.assertEqual(r['status'],'completed')
        self.assertGreater(r['speed_metrics']['nonzero_throttle_correction_samples'],0)
    def test_mock_collision_abort(self):
        rc,r,p,w=self.run_mock(collision_at=10)
        self.assertEqual(rc,2);self.assertEqual(r['collision_count'],1)
    def test_mock_stationary_abort(self):
        rc,r,p,w=self.run_mock(stationary=True)
        self.assertEqual(rc,2);self.assertIn('motion',r['abort_reason'])
    def test_mock_restore_fail(self):
        rc,r,p,w=self.run_mock(restore_fail=True)
        self.assertEqual(rc,2);self.assertEqual(r['status'],'cleanup_or_restore_failed')
    def test_compare_and_csv_audit(self):
        _,on,op,_=self.run_mock('on',measured_v_offset=-.1)
        _,off,zp,_=self.run_mock('off',measured_v_offset=-.1)
        result=tr.compare(op/'summary.json',zp/'summary.json',self.dir/'comparison.json',self.reference)
        self.assertEqual(result['status'],'comparison_complete')
        self.assertEqual(result['conclusion'],'speed_feedback_improvement_not_demonstrated')
    def test_compare_rejects_control_corruption(self):
        _,on,op,_=self.run_mock('on',measured_v_offset=-.1)
        _,off,zp,_=self.run_mock('off',measured_v_offset=-.1)
        p=zp/'calibration_samples.csv'
        with p.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
        rows[20]['command_throttle']='0.12'
        with p.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        with self.assertRaises(ValueError):tr.compare(op/'summary.json',zp/'summary.json',self.dir/'pair.json',self.reference)
    def test_compare_rejects_mismatched_hash(self):
        _,on,op,_=self.run_mock('on')
        _,off,zp,_=self.run_mock('off')
        rpath=zp/'summary.json';d=json.loads(rpath.read_text());d['model_sha256']='wrong';rpath.write_text(json.dumps(d))
        with self.assertRaises(ValueError):tr.compare(op/'summary.json',rpath,self.dir/'pair.json',self.reference)
    def test_compare_rejects_failed_run(self):
        _,on,op,_=self.run_mock('on')
        _,off,zp,_=self.run_mock('off')
        rpath=op/'summary.json';d=json.loads(rpath.read_text());d['status']='aborted';rpath.write_text(json.dumps(d))
        with self.assertRaises(ValueError):tr.compare(rpath,zp/'summary.json',self.dir/'pair.json',self.reference)

if __name__=='__main__':unittest.main()
