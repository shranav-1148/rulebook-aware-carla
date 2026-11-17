"""M6B-4 offline and mocked tests; they cannot validate real CARLA physics."""
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import carla_model_m6b4 as model
import carla_track_m6b4 as track


def fake_trial_rows(sign=1):
    rows=[]
    x,y,yaw=384.5906848222465,1.9933077714750864,3.1410605540039955
    v=0.
    for i in range(85):
        t=round(i*.05,5)
        if i:
            phase,th,st,br=model.reference_command(i)
            steer=st*sign
            if t<=1.8:v=min(1.65,t*.9)
            elif t<=2.2:v=1.62
            else:v=max(0.,1.62-(t-2.2)*1.2)
            turn=-sign*math.radians(5.33) if phase=='steer_pulse' else 0.
            dyaw=.05*v*math.tan(turn)/3.004648862774714
            xm=x+.05*v*math.cos(yaw+.5*dyaw)
            ym=y+.05*v*math.sin(yaw+.5*dyaw)
            x,y,yaw=xm,ym,model.wrap(yaw+dyaw)
        else:phase,th,steer,br='initial',0.,0.,0.
        wheel=sign*5.33 if phase=='steer_pulse' else 0.
        row={'frame':str(1000+i),'t_s':str(t),'t_sim_s':str(2000+t),
            'x_cr_m':str(x),'y_cr_m':str(y),'yaw_cr_rad':str(yaw),
            'forward_speed_mps':str(v),'lateral_speed_mps':'0',
            'vertical_speed_mps':'0','total_speed_mps':str(v),
            'wheel_front_left_deg':str(wheel),'wheel_front_right_deg':str(wheel),
            'applied_steer':str(steer),'applied_throttle':str(th),
            'applied_brake':str(br),'command_steer_normalized':str(steer),
            'command_throttle':str(th),'command_brake':str(br),
            'phase':phase,'applied_gear':'1' if i else '0',
            'applied_manual_gear_shift':'True' if i else 'False',
            'applied_hand_brake':'False','applied_reverse':'False'}
        rows.append(row)
    return rows


def trial_summary(sign):
    return {'status':'completed','mode':'manual_steering','collision_count':0,
            'cleanup_ok':True,'settings_restored':True,'samples':85,
            'throttle_motion_gate':{'passed':True},'steering_pulse_gate':{'passed':True},
            'configuration':{'mode':'manual_steering','steering_sign_command':sign,
                'estimated_wheel_geometry':{'available':True,'wheelbase_from_xy_m':3.004648862774714}},
            'preflight':{'map':'Carla/Maps/Town01','client_version':'0.9.16','server_version':'0.9.16'},
            'observation_sha256':'test-obs-hash','max_total_speed_mps':1.65,
            'final_forward_speed_mps':0.,
            'steering_diagnostics':{'final_heading_change_deg':-1.2*sign,
                                    'final_cross_track_m':-.04*sign}}


def save_fake_trial(directory, sign):
    rows=fake_trial_rows(sign)
    summary=directory/f'summary_{sign}.json'; samples=directory/f'samples_{sign}.csv'
    summary.write_text(json.dumps(trial_summary(sign)),encoding='utf-8')
    with samples.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    return summary,samples


class TestOfflineModel(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.pp,self.pc=save_fake_trial(self.root,1)
        self.np,self.nc=save_fake_trial(self.root,-1)
    def tearDown(self):self.tmp.cleanup()
    def trials(self):
        return model.read_trial(self.pp,self.pc,1),model.read_trial(self.np,self.nc,-1)
    def test_valid_pair_fit(self):
        a,b=self.trials();m=model.estimate_model(a,b)
        self.assertAlmostEqual(m['geometric_wheelbase_m'],3.004648862774714)
        self.assertAlmostEqual(m['front_wheel_signed_deg_per_normalized_steer'],66.625)
        self.assertEqual(m['carla_command_to_commonroad_steering_sign'],-1)
        self.assertFalse(m['timing_audit']['positive']['physics_wheel_angles_frame_locked'])
        self.assertLess(m['command_gain_kinematic_diagnostics']['positive']['0']['heading_rmse_deg'],2.)
    def test_invalid_sign_rejected(self):
        with self.assertRaises(ValueError):model.read_trial(self.pp,self.pc,-1)
    def test_incomplete_csv_rejected(self):
        rows=list(csv.DictReader(open(self.pc,encoding='utf-8')))
        with self.pc.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows[:-1])
        with self.assertRaises(ValueError):model.read_trial(self.pp,self.pc,1)
    def test_missing_frame_rejected(self):
        rows=list(csv.DictReader(open(self.pc,encoding='utf-8')));rows[12]['frame']='100000'
        with self.pc.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        with self.assertRaises(ValueError):model.read_trial(self.pp,self.pc,1)
    def test_bad_time_rejected(self):
        rows=list(csv.DictReader(open(self.pc,encoding='utf-8')));rows[12]['t_sim_s']='2000.7'
        with self.pc.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        with self.assertRaises(ValueError):model.read_trial(self.pp,self.pc,1)
    def test_control_gear_mismatch_rejected(self):
        rows=list(csv.DictReader(open(self.pc,encoding='utf-8')));rows[20]['applied_gear']='0'
        with self.pc.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        with self.assertRaises(ValueError):model.read_trial(self.pp,self.pc,1)
    def test_wrong_observation_rejected(self):
        b=trial_summary(-1);b['observation_sha256']='different'
        self.np.write_text(json.dumps(b))
        a,c=self.trials()
        with self.assertRaises(ValueError):model.estimate_model(a,c)
    def test_cleanup_missing_rejected(self):
        d=trial_summary(1);d['cleanup_ok']=False;self.pp.write_text(json.dumps(d))
        with self.assertRaises(ValueError):model.read_trial(self.pp,self.pc,1)
    def test_steering_sign_physics_rejected(self):
        rows=list(csv.DictReader(open(self.pc,encoding='utf-8')))
        for r in rows:
            if r['phase']=='steer_pulse':r['wheel_front_left_deg']='-5.33'
        with self.pc.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        a,b=self.trials()
        with self.assertRaises(ValueError):model.estimate_model(a,b)
    def test_main_offline_only(self):
        path=self.root/'fit.json'
        code=model.main(['--positive-summary',str(self.pp),'--positive-samples',str(self.pc),
            '--negative-summary',str(self.np),'--negative-samples',str(self.nc),'--out',str(path)])
        self.assertEqual(code,0);self.assertTrue(path.exists())
        self.assertEqual(model.main(['--positive-summary',str(self.pp),'--positive-samples',str(self.pc),
            '--negative-summary',str(self.np),'--negative-samples',str(self.nc),'--out',str(path)]),2)
    def test_model_provenance_validation(self):
        a,b=self.trials();p=self.root/'model.json'
        doc=model.estimate_model(a,b);p.write_text(json.dumps(doc))
        model.model_for_tracking(p,'test-obs-hash',a['samples_sha256'],a['summary_sha256'])
        with self.assertRaises(ValueError):model.model_for_tracking(p,'wrong',a['samples_sha256'],a['summary_sha256'])
        with self.assertRaises(ValueError):model.model_for_tracking(p,'test-obs-hash','fake',a['summary_sha256'])
    def test_no_gain_extrapolation(self):
        a,b=self.trials();m=model.estimate_model(a,b)
        self.assertAlmostEqual(m['calibrated_steer_command_abs_max'],.08)
        self.assertFalse(m['automatic_mode_validated'])


class FakeControl:
    def __init__(self,**kwargs):
        self.__dict__.update(dict(throttle=0.,steer=0.,brake=0.,hand_brake=False,reverse=False,manual_gear_shift=False,gear=0))
        self.__dict__.update(kwargs)

class FakeCarla:
    VehicleControl=FakeControl
    @staticmethod
    def Location(x=0,y=0,z=0):return NS(x=x,y=y,z=z)
    @staticmethod
    def Rotation(yaw=0):return NS(yaw=yaw)
    @staticmethod
    def Transform(loc=None,rot=None):return NS(location=loc,rotation=rot)

class FakeActor:
    id=111
    bounding_box=NS(location=NS(x=0,y=0,z=0))
    def __init__(self):self.control=FakeControl(throttle=0.,steer=0.,brake=.5,hand_brake=False,reverse=False,manual_gear_shift=False,gear=0)
    def apply_control(self,c):self.control=c
    def get_control(self):return self.control
    def set_transform(self,t):pass
    def set_autopilot(self,a):pass
    def set_simulate_physics(self,a):pass
    def destroy(self):return True

class FakeSensor:
    def listen(self,callback):self.callback=callback
    def stop(self):pass
    def destroy(self):return True

class FakeBlueprint:
    def has_attribute(self,s):return True
    def set_attribute(self,k,v):pass

class FakeWorld:
    def __init__(self):
        self.actor=FakeActor();self.sensor=FakeSensor();self.synchronous_mode=False
        self.fixed_delta_seconds=None;self.sim_t=0.;self.drive_ticks=0;self.frame=0
        self.inject_collision_at=None;self.fail_restore=False
    def get_settings(self):return NS(synchronous_mode=self.synchronous_mode,fixed_delta_seconds=self.fixed_delta_seconds)
    def apply_settings(self,settings):
        self.synchronous_mode=bool(settings.synchronous_mode)
        self.fixed_delta_seconds=settings.fixed_delta_seconds
        if self.fail_restore and not self.synchronous_mode:raise RuntimeError('Fake restore failed')
    def try_spawn_actor(self,*a):return self.actor
    def spawn_actor(self,*args,**kwargs):return self.sensor
    def get_blueprint_library(self):return NS(find=lambda name:FakeBlueprint())
    def get_actors(self):return NS(filter=lambda typ: [self.actor] if typ.startswith('vehicle') else [])
    def tick(self):
        self.frame+=1;self.sim_t+=.05
        if getattr(self.actor.control,'manual_gear_shift',False):self.drive_ticks+=1
        if self.inject_collision_at==self.drive_ticks:
            self.inject_collision_at=None
            self.sensor.callback(NS(frame=self.frame,other_actor=NS(id=999,type_id='static.test')))
        return self.frame


class TestTracking(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.pp,self.pc=save_fake_trial(self.root,1)
        self.np,self.nc=save_fake_trial(self.root,-1)
        self.pos=model.read_trial(self.pp,self.pc,1)
        self.neg=model.read_trial(self.np,self.nc,-1)
        self.cal=model.estimate_model(self.pos,self.neg)
        self.modelfile=self.root/'fit.json';self.modelfile.write_text(json.dumps(self.cal))
        self.obs={'body_center_cr_xy_m':[384.5906848222465,1.9933077714750864],
                  'yaw_cr_rad':3.1410605540039955,'sha256':'test-obs-hash','source_frame':967030}
        self.args=NS(timeout=10.,model=self.modelfile,out=self.root/'tracking')
    def tearDown(self):self.tmp.cleanup()
    def test_reference_relative_start(self):
        start={'x_cr_m':10.,'y_cr_m':5.,'yaw_cr_rad':math.pi/2}
        target={'x_cr_m':10.,'y_cr_m':6.,'yaw_cr_rad':math.pi/2+.1}
        s,lat,yaw=track.local_pose(target,start)
        self.assertAlmostEqual(s,1.);self.assertAlmostEqual(lat,0.);self.assertAlmostEqual(yaw,.1)
    def test_tracking_error_zero_for_offset_route(self):
        refstart={'x_cr_m':0.,'y_cr_m':0.,'yaw_cr_rad':0.}
        ref={'x_cr_m':1.,'y_cr_m':.05,'yaw_cr_rad':.03}
        actualstart={'x_cr_m':7.,'y_cr_m':8.,'yaw_cr_rad':math.pi/2}
        actual={'x_cr_m':7.-.05,'y_cr_m':9.,'yaw_cr_rad':math.pi/2+.03}
        e=track.tracking_error(actual,actualstart,ref,refstart)
        self.assertLess(e['position_error_m'],1e-9)
        self.assertAlmostEqual(e['heading_error_deg'],0.)
    def test_feedback_steering_correct_sign(self):
        ref={'command_throttle':'0.20','command_brake':'0','command_steer_normalized':'0'}
        err={'lateral_error_m':.02,'heading_error_deg':1.}
        res=track.command_from_error(ref,err,self.cal)
        self.assertGreater(res[1],0.)
        err={'lateral_error_m':-.02,'heading_error_deg':-1.}
        self.assertLess(track.command_from_error(ref,err,self.cal)[1],0.)
    def test_steering_command_never_exceeds_tested_range(self):
        ref={'command_throttle':'0.20','command_brake':'0','command_steer_normalized':'0.08'}
        e={'lateral_error_m':.1,'heading_error_deg':3.}
        self.assertLessEqual(track.command_from_error(ref,e,self.cal)[1],.08)
    def test_bad_model_sign_rejected(self):
        c=dict(self.cal,carla_command_to_commonroad_steering_sign=1)
        r=self.pos['rows'][22];e={'lateral_error_m':0.,'heading_error_deg':0.}
        with self.assertRaises(ValueError):track.command_from_error(r,e,c)
    def test_tracking_error_gate(self):
        base={'position_error_m':0.,'heading_error_deg':0.,'total_speed_mps':1.,
              'forward_speed_mps':1.,'distance_from_start_m':1.,'cross_track_m':0.,
              'heading_from_start_deg':0.,'vertical_speed_mps':0.,
              'wheel_front_left_deg':0.,'wheel_front_right_deg':0.}
        self.assertFalse(track.tracking_abort_reasons(base))
        for k,v in (('position_error_m',.21),('heading_error_deg',4.1),('total_speed_mps',2.2),
                    ('cross_track_m',.26),('heading_from_start_deg',5.1),('vertical_speed_mps',1.),
                    ('distance_from_start_m',4.1),('wheel_front_left_deg',11.)):
            self.assertTrue(track.tracking_abort_reasons(dict(base,**{k:v})))
    def test_applied_control_gate(self):
        row=dict(applied_throttle=.2,applied_steer=.08,applied_brake=0.,
                 applied_hand_brake=False,applied_reverse=False,applied_manual_gear_shift=True,applied_gear=1)
        track.check_expected_applied(row,.2,.08,0.)
        for bad in (dict(row,applied_gear=0),dict(row,applied_steer=0.),dict(row,applied_manual_gear_shift=False)):
            with self.assertRaises(RuntimeError):track.check_expected_applied(bad,.2,.08,0.)
    def test_offline_default_never_connects_carla(self):
        with patch('carla_track_m6b4.b.read_stable_observation',return_value=self.obs), \
             patch('carla_track_m6b4.b.preflight',side_effect=AssertionError('must remain offline')):
            self.assertEqual(track.main(['--observation','ignored',
                '--positive-summary',str(self.pp),'--positive-samples',str(self.pc),
                '--model',str(self.modelfile)]),0)
    def test_model_hash_mismatch_rejected_before_server_access(self):
        broken=dict(self.cal,observation_sha256='wrong')
        self.modelfile.write_text(json.dumps(broken))
        with patch('carla_track_m6b4.b.read_stable_observation',return_value=self.obs), \
             patch('carla_track_m6b4.b.preflight',side_effect=AssertionError('must remain offline')):
            self.assertEqual(track.main(['--observation','ignored',
                '--positive-summary',str(self.pp),'--positive-samples',str(self.pc),
                '--model',str(self.modelfile)]),2)

    def test_requires_both_drive_switches(self):
        for argv in (['--execute-track'],['--confirm-drive']):
            with self.assertRaises(SystemExit) as exc:track.main(argv)
            self.assertEqual(exc.exception.code,2)
    def mock_world_run(self, stationary=False, collision_at=None, restore_fail=False):
        world=FakeWorld();world.inject_collision_at=collision_at;world.fail_restore=restore_fail
        reference=self.pos['rows']
        def sample(w,a,c,bb,expected_frame=None):
            src=reference[0 if stationary else min(w.drive_ticks,84)]
            return {'frame':w.frame,'t_sim_s':w.sim_t,
                    'x_cr_m':float(src['x_cr_m']),'y_cr_m':float(src['y_cr_m']),
                    'yaw_cr_rad':float(src['yaw_cr_rad']),
                    'forward_speed_mps':0. if stationary else float(src['forward_speed_mps']),
                    'lateral_speed_mps':0.,'vertical_speed_mps':0.,
                    'total_speed_mps':0. if stationary else float(src['forward_speed_mps']),
                    'wheel_front_left_deg':0. if stationary else float(src['wheel_front_left_deg']),
                    'wheel_front_right_deg':0. if stationary else float(src['wheel_front_right_deg'])}
        wp=NS(location=NS(z=0.))
        with patch('carla_track_m6b4.b.preflight',return_value=(FakeCarla,None,world,FakeBlueprint(),wp,{'mock':True})),\
             patch('carla_track_m6b4.b._get_tick',side_effect=lambda w,timeout_s=None: w.tick()),\
             patch('carla_track_m6b4.b.sample_actor',side_effect=sample):
            code=track.execute(self.args,self.obs,self.pos,self.cal)
        data=json.loads((self.args.out/'summary.json').read_text())
        return code,data,world
    def test_mock_tracking_completed_and_cleanup(self):
        code,result,world=self.mock_world_run()
        self.assertEqual(code,0);self.assertEqual(result['status'],'completed')
        self.assertEqual(result['collision_count'],0)
        self.assertTrue(result['cleanup_ok']);self.assertTrue(result['settings_restored'])
        self.assertEqual(result['samples'],85)
        self.assertLessEqual(result['max_abs_steer_command'],.08)
        self.assertFalse(world.synchronous_mode)
    def test_mock_no_motion_aborts_and_cleans(self):
        code,result,world=self.mock_world_run(stationary=True)
        self.assertEqual(code,2);self.assertEqual(result['status'],'aborted')
        self.assertIn('No longitudinal motion',result['abort_reason'])
        self.assertTrue(result['cleanup_ok']);self.assertFalse(world.synchronous_mode)
    def test_mock_collision_aborts_and_cleans(self):
        code,result,world=self.mock_world_run(collision_at=10)
        self.assertEqual(code,2);self.assertEqual(result['status'],'aborted')
        self.assertEqual(result['collision_count'],1)
        self.assertTrue(result['cleanup_ok']);self.assertFalse(world.synchronous_mode)
    def test_mock_failed_restore_cannot_report_success(self):
        code,result,world=self.mock_world_run(restore_fail=True)
        self.assertEqual(code,2);self.assertEqual(result['status'],'cleanup_or_restore_failed')
        self.assertFalse(result['settings_restored'])
    def test_result_overwrite_refused_before_connecting_to_carla(self):
        self.args.out.mkdir();(self.args.out/'summary.json').write_text('old')
        with patch('carla_track_m6b4.b.read_stable_observation',return_value=self.obs), \
             patch('carla_track_m6b4.b.preflight',side_effect=RuntimeError('Must not connect')):
            result=track.main(['--execute-track','--confirm-drive','--observation','ignored',
                '--positive-summary',str(self.pp),'--positive-samples',str(self.pc),
                '--model',str(self.modelfile),'--out',str(self.args.out)])
            self.assertEqual(result,2)
        self.assertEqual((self.args.out/'summary.json').read_text(),'old')

if __name__=='__main__':unittest.main()
