"""Offline/mock tests. They do NOT validate CARLA vehicle physics or actual control safety."""
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from copy import copy

import carla_calibrate_m6b3 as m


def obj(**kwargs):
    return NS(**kwargs)


def stable_json():
    return {'stability_capture_mode':'passive_multi_frame', 'cleanup_ok':True,
            'stability':{'latest':{'accepted':True}},
            'checks':{k:True for k in ('ego_pose_matches_nominal_start','obstacle_center_matches_offline_location',
              'ego_within_route_offset_bound','ego_within_route_heading_bound',
              'initial_wheel_steering_near_zero','small_lateral_velocity')},
            'server':{'map':'Carla/Maps/Town01','carla_client_version':'0.9.16','carla_server_version':'0.9.16'},
            'ego':{'body_center_cr_xy_m':[384.59083741,1.99333765],
                   'yaw_cr_rad':3.1410602876,'longitudinal_speed_mps':-2.394e-6,
                   'lateral_speed_mps':1.6e-5,'velocity_carla_xyz_mps':[0,0,0.00001]},
            'snapshot':{'frame':967030}}


def default_obs():
    return {'body_center_cr_xy_m':[384.59083741,1.99333765],
            'yaw_cr_rad':3.1410602876,'source_frame':967030,'sha256':'fakehash'}


class TestPure(unittest.TestCase):
    def test_finite_nan(self):
        with self.assertRaises(ValueError):m.finite('v',float('nan'))
    def test_wrap(self):
        self.assertAlmostEqual(abs(m.wrap(3*math.pi)),math.pi)
    def test_body_center_inverse(self):
        b=obj(x=.3,y=-.02)
        for yaw in (-math.pi,0,math.pi/2,.3):
            x,y=m.actor_origin_for_center(12,-9,yaw,b)
            cx,cy=m.center_from_actor(x,y,yaw,b)
            self.assertAlmostEqual(cx,12)
            self.assertAlmostEqual(cy,-9)
    def test_parse_valid_observation(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'obs.json';p.write_text(json.dumps(stable_json()))
            data=m.read_stable_observation(p)
            self.assertEqual(data['source_frame'],967030)
    def test_reject_unstable(self):
        with tempfile.TemporaryDirectory() as t:
            doc=stable_json();doc['stability']['latest']['accepted']=False
            p=Path(t)/'obs.json';p.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):m.read_stable_observation(p)
    def test_reject_wrong_town(self):
        with tempfile.TemporaryDirectory() as t:
            doc=stable_json();doc['server']['map']='Carla/Maps/Town02'
            p=Path(t)/'obs.json';p.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):m.read_stable_observation(p)
    def test_reject_mismatched_start(self):
        with tempfile.TemporaryDirectory() as t:
            doc=stable_json();doc['ego']['body_center_cr_xy_m'][0]=100
            p=Path(t)/'obs.json';p.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):m.read_stable_observation(p)
    def test_reject_cleanup_failed(self):
        with tempfile.TemporaryDirectory() as t:
            doc=stable_json();doc['cleanup_ok']=False
            p=Path(t)/'obs.json';p.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):m.read_stable_observation(p)
    def test_reject_unsafe_stable_speed(self):
        with tempfile.TemporaryDirectory() as t:
            doc=stable_json();doc['ego']['longitudinal_speed_mps']=-.3
            p=Path(t)/'obs.json';p.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):m.read_stable_observation(p)
    def test_motion_gate_zero_is_failure(self):
        row={'phase':'throttle','forward_speed_mps':0.,'distance_from_start_m':0.}
        self.assertFalse(m.motion_during_throttle([row])['passed'])

    def test_motion_gate_real_response_passes(self):
        row={'phase':'throttle','forward_speed_mps':.5,'distance_from_start_m':.15}
        self.assertTrue(m.motion_during_throttle([row])['passed'])

    def test_readback_control_flags(self):
        c=obj(throttle=.25,steer=0.,brake=0.,hand_brake=False,reverse=False,manual_gear_shift=False,gear=1)
        r=m.applied_control_readback(obj(get_control=lambda: c))
        self.assertEqual(r['applied_gear'],1)
        self.assertFalse(r['applied_hand_brake'])

    def test_straight_command_phases(self):
        self.assertEqual(m.plan_command('straight',0)[0],'throttle')
        self.assertEqual(m.plan_command('straight',1.5)[0],'coast')
        self.assertEqual(m.plan_command('straight',2.0)[0],'brake')
        self.assertLessEqual(m.plan_command('straight',0)[1],.25)
    def test_manual_first_uses_identical_throttle_profile(self):
        for t in (0, .5, 1.45, 1.5, 2, 4):
            self.assertEqual(m.plan_command('manual_first',t),m.plan_command('straight',t))
    def test_manual_first_control_uses_gear_one_without_more_throttle(self):
        auto=m.commanded_vehicle_control(obj(VehicleControl=MockWorld.Control),'straight',.25,0.,0.)
        manual=m.commanded_vehicle_control(obj(VehicleControl=MockWorld.Control),'manual_first',.25,0.,0.)
        self.assertFalse(auto.manual_gear_shift)
        self.assertEqual(auto.gear,0)
        self.assertTrue(manual.manual_gear_shift)
        self.assertEqual(manual.gear,1)
        self.assertEqual(manual.throttle,auto.throttle)
        self.assertFalse(manual.hand_brake)
        self.assertFalse(manual.reverse)
    def test_manual_first_requires_explicit_drive_confirmation(self):
        with self.assertRaises(SystemExit) as e:m.main(['--execute-manual-first'])
        self.assertEqual(e.exception.code,2)
    def test_manual_first_not_accepted_as_automatic_steering_baseline(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'manual.json';p.write_text(json.dumps(dict(mode='manual_first',status='completed',
                     cleanup_ok=True,settings_restored=True,collision_count=0,
                     observation_sha256='fakehash',max_displacement_m=1.,max_total_speed_mps=1.,
                     throttle_motion_gate={'passed':True})))
            with self.assertRaises(ValueError):m.validate_baseline(p,'fakehash')
    def test_wheelbase_geometry_world_centimeters(self):
        p=obj(wheels=[obj(position=obj(x=38297.2,y=-116.0)),
                       obj(position=obj(x=38297.3,y=-282.7)),
                       obj(position=obj(x=38597.7,y=-115.9)),
                       obj(position=obj(x=38597.8,y=-282.6))])
        self.assertAlmostEqual(m.estimated_wheel_geometry(p)['wheelbase_from_xy_m'],3.005,places=2)
    def test_steering_command_bounded(self):
        for sign in (-1,1):
            c=m.plan_command('steering',1.2,sign)
            self.assertEqual(c[0],'steer_pulse')
            self.assertAlmostEqual(c[2],sign*.08)

    def test_manual_steering_reuses_bounded_original_pulse(self):
        for t in (0., .8,1.15,1.5,1.9,2.5,4.):
            self.assertEqual(m.plan_command('manual_steering',t),m.plan_command('steering',t))
    def test_manual_steering_explicit_first_gear(self):
        c=m.commanded_vehicle_control(obj(VehicleControl=MockWorld.Control),'manual_steering',.2,.08,0.)
        self.assertTrue(c.manual_gear_shift)
        self.assertEqual(c.gear,1)
        self.assertEqual(c.steer,.08)
        self.assertEqual(c.throttle,.2)
    def test_manual_steering_requires_confirm_and_baseline(self):
        for argv in (['--execute-manual-steering'],['--execute-manual-steering','--confirm-drive']):
            with self.assertRaises(SystemExit) as e:m.main(argv)
            self.assertEqual(e.exception.code,2)
    def test_manual_steering_baseline_accepts_manual_first_not_auto(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'s.json'
            doc=dict(mode='manual_first',status='completed',cleanup_ok=True,settings_restored=True,
                 collision_count=0,observation_sha256='fakehash',max_displacement_m=1.838,
                 max_total_speed_mps=1.614,throttle_motion_gate={'passed':True})
            path.write_text(json.dumps(doc))
            self.assertEqual(m.validate_baseline(path,'fakehash',expected_mode='manual_first')['status'],'completed')
            with self.assertRaises(ValueError):m.validate_baseline(path,'fakehash',expected_mode='straight')
    def test_manual_steering_baseline_rejects_bad_cleanup(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'s.json'
            p.write_text(json.dumps(dict(mode='manual_first',status='completed',cleanup_ok=False,settings_restored=True,
                 collision_count=0,observation_sha256='fakehash',max_displacement_m=1.,max_total_speed_mps=1.,
                 throttle_motion_gate={'passed':True})))
            with self.assertRaises(ValueError):m.validate_baseline(p,'fakehash',expected_mode='manual_first')
    def test_manual_steering_readback_gate(self):
        row=dict(applied_throttle=.2,applied_steer=.08,applied_brake=0.,applied_hand_brake=False,
                 applied_reverse=False,applied_manual_gear_shift=True,applied_gear=1)
        m.manual_steering_control_check(row,'steer_pulse',.2,.08,0.)
        for change in ({'applied_steer':0.},{'applied_gear':0},{'applied_manual_gear_shift':False},
                       {'applied_brake':.5},{'applied_hand_brake':True}):
            wrong=dict(row,**change)
            with self.assertRaises(RuntimeError):m.manual_steering_control_check(wrong,'steer_pulse',.2,.08,0.)
    def test_manual_steering_gate_requires_yaw_and_wheel(self):
        row=dict(phase='steer_pulse',forward_speed_mps=1.,wheel_front_left_deg=1.6,
                 wheel_front_right_deg=1.5,heading_from_start_deg=.2,applied_steer=.08)
        self.assertTrue(m.steering_pulse_gate([row])['passed'])
        self.assertFalse(m.steering_pulse_gate([dict(row,heading_from_start_deg=0.)])['passed'])
        self.assertFalse(m.steering_pulse_gate([dict(row,wheel_front_left_deg=0.,wheel_front_right_deg=0.)])['passed'])
        self.assertFalse(m.steering_pulse_gate([dict(row,forward_speed_mps=0.)])['passed'])

    def test_unknown_command_rejected(self):
        with self.assertRaises(ValueError):m.plan_command('nope',0)
    def test_enrich_cross_track(self):
        start={'x_cr_m':0.,'y_cr_m':0.,'yaw_cr_rad':0.}
        row={'x_cr_m':2.,'y_cr_m':.2,'yaw_cr_rad':0.}
        r=m.enrich(row,start,('test',0.,0.,0.),1.)
        self.assertAlmostEqual(r['along_track_m'],2.)
        self.assertAlmostEqual(r['cross_track_m'],.2)
    def test_speed_limit_fails(self):
        row={'total_speed_mps':3.6,'forward_speed_mps':3.5,'distance_from_start_m':1.,
             'cross_track_m':0.,'heading_from_start_deg':0.,'vertical_speed_mps':0.}
        self.assertIn('speed/reverse bound',m.evaluate_limits(row))
    def test_displacement_limit_fails(self):
        row={'total_speed_mps':1.,'forward_speed_mps':1.,'distance_from_start_m':8.1,
             'cross_track_m':0.,'heading_from_start_deg':0.,'vertical_speed_mps':0.}
        self.assertIn('displacement bound',m.evaluate_limits(row))
    def test_kinematic_sign_hypothesis(self):
        r=[]
        for j in range(3):
            r.append({'t_s':j*.1,'x_cr_m':j*.1,'y_cr_m':0.,'yaw_cr_rad':0.,
                 'forward_speed_mps':1.,'wheel_front_left_deg':0.,'wheel_front_right_deg':0.})
        res=m.kinematic_model_comparison(r)
        self.assertAlmostEqual(res['sign_hypothesis_rmse_m']['+'],0.)
        self.assertAlmostEqual(res['sign_hypothesis_rmse_m']['-'],0.)
    def test_baseline_reject_wrong_source(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'s.json'; p.write_text(json.dumps(dict(mode='straight',status='completed',
                  cleanup_ok=True,settings_restored=True,collision_count=0,
                  observation_sha256='other',max_displacement_m=1.,max_total_speed_mps=1.)))
            with self.assertRaises(ValueError):m.validate_baseline(p,'fakehash')
    def test_legacy_completed_but_stuck_baseline_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'old.json'
            p.write_text(json.dumps(dict(mode='straight',status='completed',
                cleanup_ok=True,settings_restored=True,collision_count=0,
                observation_sha256='fakehash',max_displacement_m=2.15e-6,
                max_total_speed_mps=.00179)))
            with self.assertRaises(ValueError):m.validate_baseline(p,'fakehash')

    def test_baseline_accept_valid(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'s.json'; p.write_text(json.dumps(dict(mode='straight',status='completed',
                  cleanup_ok=True,settings_restored=True,collision_count=0,
                  observation_sha256='fakehash',max_displacement_m=1.,max_total_speed_mps=1.,
                  throttle_motion_gate={'passed':True})))
            self.assertEqual(m.validate_baseline(p,'fakehash')['status'],'completed')
    def test_save_never_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'out'
            m.save_results(p,[],{'status':'test'})
            with self.assertRaises(FileExistsError):m.save_results(p,[],{'status':'test'})
    def test_no_drive_by_default(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'obs.json';p.write_text(json.dumps(stable_json()))
            self.assertEqual(m.main(['--observation',str(p)]),0)
    def test_confirm_without_drive_rejected(self):
        with self.assertRaises(SystemExit) as e:m.main(['--confirm-drive'])
        self.assertEqual(e.exception.code,2)
    def test_drive_without_confirm_rejected(self):
        with self.assertRaises(SystemExit) as e:m.main(['--execute-straight'])
        self.assertEqual(e.exception.code,2)
    def test_steering_without_baseline_rejected(self):
        with self.assertRaises(SystemExit) as e:m.main(['--execute-steering','--confirm-drive'])
        self.assertEqual(e.exception.code,2)


class TestPreflight(unittest.TestCase):
    def stub(self, vehicles=0, walkers=0, props=0, sync=False, lane_distance=0.,
             mesh_positions=(), unexpected_props=(), broken_mesh=False):
        class FakeProp:
            def __init__(self, ident, typ, x, y, broken=False):
                self.id,self.type_id=ident,typ
                self.x,self.y,self.broken=x,y,broken
            def get_transform(self):return obj(location=obj(x=self.x,y=self.y,z=0.))
            @property
            def bounding_box(self):
                if self.broken:raise RuntimeError('mock unavailable geometry')
                pos=self.get_transform().location
                return obj(get_world_vertices=lambda pose:[obj(x=pos.x+sx,y=pos.y+sy,z=sz)
                       for sx in (-.2,.2) for sy in (-.2,.2) for sz in (0.,.4)])
        actual_props = ([object() for _ in range(props)] +
                        [FakeProp(i,'static.prop.mesh',x,y) for i,(x,y) in enumerate(mesh_positions,100)] +
                        [FakeProp(i,'static.prop.constructioncone',x,y) for i,(x,y) in enumerate(unexpected_props,200)] +
                        ([FakeProp(300,'static.prop.mesh',0.,0.,True)] if broken_mesh else []))
        class Actors:
            def filter(self, pattern):
                if pattern.startswith('vehicle'):return [object() for _ in range(vehicles)]
                if pattern.startswith('walker'):return [object() for _ in range(walkers)]
                return actual_props
        class BP:
            id = 'vehicle.tesla.model3'
        class Map:
            name = 'Carla/Maps/Town01'
            def get_waypoint(self, location, project_to_road=True, lane_type=None):
                return obj(transform=obj(location=obj(x=location.x,y=location.y+lane_distance,z=0.),
                                         rotation=obj(yaw=-180.)))
        class World:
            def get_map(self):return Map()
            def get_blueprint_library(self):return obj(find=lambda k:BP())
            def get_actors(self):return Actors()
            def get_settings(self):return obj(synchronous_mode=sync)
        class Client:
            def __init__(self,*a):pass
            def set_timeout(self,n):pass
            def get_client_version(self):return '0.9.16'
            def get_server_version(self):return '0.9.16'
            def get_world(self):return World()
        fake=obj(Client=Client,Location=lambda x,y,z:obj(x=x,y=y,z=z),LaneType=obj(Driving='Driving'))
        return fake
    def run_preflight(self,fake,drive=False):
        import sys
        with patch.dict(sys.modules, {'carla':fake}):
            return m.preflight(obj(host='127.0.0.1',port=2000,timeout=10.),default_obs(),drive)
    def test_read_only_inspection_accepts_empty_async_world(self):
        ret=self.run_preflight(self.stub())
        self.assertEqual(ret[-1]['existing_vehicles'],0)
        self.assertEqual(len(ret[-1]['ahead_checks']),4)
    def test_drive_refuses_occupied_world(self):
        with self.assertRaises(RuntimeError):self.run_preflight(self.stub(vehicles=1),drive=True)
        with self.assertRaises(RuntimeError):self.run_preflight(self.stub(walkers=1),drive=True)
        with self.assertRaises(RuntimeError):self.run_preflight(self.stub(props=1),drive=True)
    def test_drive_accepts_115_distant_mesh_props(self):
        far = [(0.,0.)]*115  # far from the validated x=384.59 CARLA start
        info=self.run_preflight(self.stub(mesh_positions=far),drive=True)[-1]
        self.assertEqual(info['existing_props'],115)
        screen=info['static_prop_screen']
        self.assertTrue(screen['clear_for_calibration'])
        self.assertEqual(screen['permitted_distant_mesh_count'],115)
        self.assertEqual(screen['nearby_count'],0)

    def test_drive_rejects_mesh_intersecting_corridor(self):
        with self.assertRaisesRegex(RuntimeError,'nearby=1'):
            self.run_preflight(self.stub(mesh_positions=[(380.,-1.993)]),drive=True)

    def test_drive_rejects_unexpected_prop_even_if_far(self):
        with self.assertRaisesRegex(RuntimeError,'unexpected_types=1'):
            self.run_preflight(self.stub(unexpected_props=[(0.,0.)]),drive=True)

    def test_drive_rejects_uninspectable_mesh(self):
        with self.assertRaisesRegex(RuntimeError,'uninspectable=1'):
            self.run_preflight(self.stub(broken_mesh=True),drive=True)

    def test_inspect_server_reports_nearby_without_driving(self):
        screen=self.run_preflight(self.stub(mesh_positions=[(380.,-1.993)]),drive=False)[-1]['static_prop_screen']
        self.assertFalse(screen['clear_for_calibration'])
        self.assertEqual(screen['nearby_count'],1)

    def test_drive_refuses_already_synchronous(self):
        with self.assertRaises(RuntimeError):self.run_preflight(self.stub(sync=True),drive=True)
    def test_preflight_rejects_misaligned_lane(self):
        with self.assertRaises(RuntimeError):self.run_preflight(self.stub(lane_distance=.6))


class MockWorld:
    class Location:
        def __init__(self,x=0.,y=0.,z=0.):self.x,self.y,self.z=x,y,z
    class Rotation:
        def __init__(self,yaw=0.):self.yaw=yaw
    class Transform:
        def __init__(self,location=None,rotation=None):
            self.location=location if location is not None else MockWorld.Location()
            self.rotation=rotation if rotation is not None else MockWorld.Rotation()
    class Control:
        def __init__(self,throttle=0.,steer=0.,brake=0.,hand_brake=False,reverse=False,manual_gear_shift=False,gear=0):
            self.throttle,self.steer,self.brake=throttle,steer,brake
            self.hand_brake,self.reverse,self.manual_gear_shift,self.gear=hand_brake,reverse,manual_gear_shift,gear
    def __init__(self,collision=False,fail_after=None,stuck=False,brakes_stuck=False,requires_first_gear=False,steering_stuck=False):
        self.settings=obj(synchronous_mode=False,fixed_delta_seconds=None)
        self.count=0;self.actor=None;self.sensor=None
        self.force_collision=collision;self.fail_after=fail_after
        self.stuck=stuck;self.brakes_stuck=brakes_stuck;self.requires_first_gear=requires_first_gear
        self.steering_stuck=steering_stuck
        self.carla=obj(Location=self.Location,Rotation=self.Rotation,Transform=self.Transform,
                       VehicleControl=self.Control,VehicleWheelLocation=obj(FL_Wheel=0,FR_Wheel=1))
    def get_settings(self):return copy(self.settings)
    def apply_settings(self,s):self.settings=copy(s)
    def try_spawn_actor(self,bp,pose):
        self.actor=MockVehicle(pose,stuck=self.stuck,brakes_stuck=self.brakes_stuck,requires_first_gear=self.requires_first_gear,steering_stuck=self.steering_stuck)
        return self.actor
    def get_blueprint_library(self):return obj(find=lambda name:obj(id=name))
    def spawn_actor(self,bp,transform,attach_to=None):
        self.sensor=MockSensor()
        return self.sensor
    def tick(self,timeout=10.):
        self.count+=1
        if self.fail_after is not None and self.count>=self.fail_after:
            raise RuntimeError('mock tick failure')
        if self.actor is not None:
            self.actor.update(self.settings.fixed_delta_seconds or .05)
            if self.force_collision and self.count==32 and self.sensor and self.sensor.listener:
                self.sensor.listener(obj(frame=self.count,other_actor=obj(id=5,type_id='static.prop.test')))
        return self.count
    def get_snapshot(self):
        actor=self.actor
        class SnapActor:
            def get_transform(self):return actor.pose
            def get_velocity(self):return obj(x=actor.v*math.cos(math.radians(actor.pose.rotation.yaw)),
                                               y=actor.v*math.sin(math.radians(actor.pose.rotation.yaw)),z=0.)
            def get_acceleration(self):return obj(x=0.,y=0.,z=0.)
        return obj(frame=self.count,find=lambda id:SnapActor() if id==actor.id else None,
                   timestamp=obj(elapsed_seconds=self.count*.05))


class MockVehicle:
    def __init__(self,pose,stuck=False,brakes_stuck=False,requires_first_gear=False,steering_stuck=False):
        self.id=10;self.pose=pose;self.v=0.
        self.stuck=stuck;self.brakes_stuck=brakes_stuck;self.requires_first_gear=requires_first_gear
        self.steering_stuck=steering_stuck
        self.bounding_box=obj(location=obj(x=.029,y=0.,z=.736))
        self.control=MockWorld.Control();self.controls=[];self.destroyed=False
    def set_transform(self,t):self.pose=t
    def set_autopilot(self,x):pass
    def set_simulate_physics(self,x):pass
    def apply_control(self,c):self.control=c;self.controls.append(c)
    def get_control(self):
        if self.brakes_stuck and self.control.throttle>0:
            c=copy(self.control);c.brake=.8;return c
        return self.control
    def get_wheel_steer_angle(self,idx):return 0. if self.steering_stuck else self.control.steer*20.
    def get_physics_control(self):return obj(wheels=[obj(max_steer_angle=60.),obj(max_steer_angle=60.)],mass=2000.)
    def update(self,dt):
        if self.stuck:return
        c=self.control
        if self.requires_first_gear and not (c.manual_gear_shift and c.gear==1):return
        self.v=max(0.,self.v+dt*(3*c.throttle-5*c.brake-.02))
        yaw=math.radians(self.pose.rotation.yaw)
        yaw+=self.v*math.tan(math.radians(0. if self.steering_stuck else 20*c.steer))/2.7*dt
        self.pose.rotation.yaw=math.degrees(yaw)
        self.pose.location.x+=dt*self.v*math.cos(yaw)
        self.pose.location.y+=dt*self.v*math.sin(yaw)
    def destroy(self):self.destroyed=True;return True


class MockSensor:
    def __init__(self):self.listener=None;self.destroyed=False
    def listen(self,fn):self.listener=fn
    def stop(self):pass
    def destroy(self):self.destroyed=True;return True


class TestLifecycle(unittest.TestCase):
    def run_mock(self,mode,collision=False,fail_after=None,stuck=False,brakes_stuck=False,requires_first_gear=False,steering_stuck=False,steer_sign=1):
        world=MockWorld(collision=collision,fail_after=fail_after,stuck=stuck,brakes_stuck=brakes_stuck,requires_first_gear=requires_first_gear,steering_stuck=steering_stuck)
        with tempfile.TemporaryDirectory() as t:
            args=obj(out=Path(t)/'run',timeout=10.,steer_sign=steer_sign)
            pre=obj(location=obj(z=0.))
            with patch.object(m,'preflight',return_value=(world.carla,None,world,obj(id='vehicle.tesla.model3',
                 has_attribute=lambda k:False),pre,dict(map='Carla/Maps/Town01', static_prop_screen=dict(total=0, permitted_distant_mesh_count=0, nearby_count=0, unexpected_type_count=0, uninspectable_count=0, clear_for_calibration=True)))):
                rc=m.drive(args,default_obs(),mode)
            report=json.loads((args.out/'summary.json').read_text())
            csv_present=(args.out/'calibration_samples.csv').exists()
            return world,rc,report,csv_present
    def test_straight_finishes_restores_cleanup(self):
        world,rc,report,csv_ok=self.run_mock('straight')
        self.assertEqual(rc,0)
        self.assertEqual(report['status'],'completed')
        self.assertTrue(report['cleanup_ok'])
        self.assertTrue(report['settings_restored'])
        self.assertFalse(world.settings.synchronous_mode)
        self.assertTrue(world.actor.destroyed)
        self.assertTrue(world.sensor.destroyed)
        self.assertTrue(csv_ok)
        self.assertGreater(report['max_displacement_m'],.5)
        self.assertFalse(report['rulebook_replanning_enabled'])
        self.assertFalse(report['obstacle_spawned'])
    def test_manual_first_executes_without_affecting_auto_baseline(self):
        world,rc,report,csv_ok=self.run_mock('manual_first',requires_first_gear=True)
        self.assertEqual(rc,0)
        self.assertEqual(report['status'],'completed')
        self.assertEqual(report['mode'],'manual_first')
        self.assertTrue(report['cleanup_ok'])
        self.assertTrue(report['settings_restored'])
        self.assertTrue(report['configuration']['gear_diagnostic'])
        self.assertTrue(report['applied_control_diagnostics']['first_throttle_readback']['applied_manual_gear_shift'])
        self.assertEqual(report['applied_control_diagnostics']['observed_gears_during_throttle'],[1])
        self.assertFalse(report['rulebook_replanning_enabled'])
    def test_auto_control_cannot_move_manual_only_mock_vehicle(self):
        world,rc,report,csv_ok=self.run_mock('straight',requires_first_gear=True)
        self.assertEqual(rc,2)
        self.assertFalse(report['throttle_motion_gate']['passed'])
    def test_manual_first_still_fails_stuck_vehicle(self):
        world,rc,report,csv_ok=self.run_mock('manual_first',stuck=True)
        self.assertEqual(rc,2)
        self.assertIn('No measurable acceleration',report['abort_reason'])
        self.assertTrue(report['cleanup_ok'])
    def test_steering_finishes(self):
        world,rc,report,csv_ok=self.run_mock('steering')
        self.assertEqual(rc,0)
        self.assertEqual(report['status'],'completed')
        self.assertGreater(report['max_abs_heading_change_deg'],0.)

    def test_manual_steering_completes_with_measured_response(self):
        world,rc,report,csv_ok=self.run_mock('manual_steering',requires_first_gear=True)
        self.assertEqual(rc,0)
        self.assertEqual(report['status'],'completed')
        self.assertTrue(report['throttle_motion_gate']['passed'])
        self.assertTrue(report['steering_pulse_gate']['passed'])
        self.assertGreater(report['steering_pulse_gate']['peak_abs_heading_change_deg'],.1)
        self.assertEqual(report['steering_diagnostics']['requested_steer_sign'],1)
        self.assertTrue(report['cleanup_ok'])
        self.assertTrue(report['settings_restored'])
        self.assertTrue(csv_ok)
        self.assertFalse(report['rulebook_replanning_enabled'])
        self.assertFalse(report['obstacle_spawned'])
    def test_manual_steering_negative_sign_produces_opposite_yaw(self):
        _,r1,rep1,_=self.run_mock('manual_steering',requires_first_gear=True,steer_sign=1)
        _,r2,rep2,_=self.run_mock('manual_steering',requires_first_gear=True,steer_sign=-1)
        self.assertEqual((r1,r2),(0,0))
        self.assertLess(rep1['steering_diagnostics']['final_heading_change_deg']*rep2['steering_diagnostics']['final_heading_change_deg'],0)
    def test_manual_steering_no_gear_cannot_move(self):
        world,rc,report,csv_ok=self.run_mock('manual_steering',requires_first_gear=True,stuck=True)
        self.assertEqual(rc,2)
        self.assertIn('No measurable manual-first acceleration',report['abort_reason'])
        self.assertTrue(report['cleanup_ok'])
    def test_manual_steering_no_wheel_turn_aborts(self):
        world,rc,report,csv_ok=self.run_mock('manual_steering',requires_first_gear=True,steering_stuck=True)
        self.assertEqual(rc,2)
        self.assertIn('No measurable steering response',report['abort_reason'])
        self.assertFalse(report['steering_pulse_gate']['passed'])
        self.assertTrue(report['settings_restored'])
        self.assertTrue(report['cleanup_ok'])
    def test_manual_steering_collision_aborts(self):
        world,rc,report,csv_ok=self.run_mock('manual_steering',requires_first_gear=True,collision=True)
        self.assertEqual(rc,2)
        self.assertEqual(report['status'],'aborted')
        self.assertGreater(report['collision_count'],0)
        self.assertTrue(world.actor.destroyed)
        self.assertFalse(world.settings.synchronous_mode)

    def test_stuck_vehicle_fails_not_completed(self):
        world,rc,report,csv_ok=self.run_mock('straight',stuck=True)
        self.assertEqual(rc,2)
        self.assertEqual(report['status'],'aborted')
        self.assertIn('No measurable acceleration',report['abort_reason'])
        self.assertFalse(report['throttle_motion_gate']['passed'])
        self.assertTrue(report['cleanup_ok'])
        self.assertTrue(report['settings_restored'])
        self.assertEqual(report['applied_control_diagnostics']['last_throttle_readback']['applied_throttle'],.25)

    def test_control_override_is_detected(self):
        world,rc,report,csv_ok=self.run_mock('straight',brakes_stuck=True)
        self.assertEqual(rc,2)
        self.assertIn('does not match applied CARLA control',report['abort_reason'])
        self.assertTrue(report['cleanup_ok'])
        self.assertTrue(report['settings_restored'])

    def test_collision_aborts_and_cleans(self):
        world,rc,report,csv_ok=self.run_mock('straight',collision=True)
        self.assertEqual(rc,2)
        self.assertEqual(report['status'],'aborted')
        self.assertGreater(report['collision_count'],0)
        self.assertTrue(report['cleanup_ok'])
        self.assertTrue(world.actor.destroyed)
        self.assertFalse(world.settings.synchronous_mode)
    def test_tick_failure_aborts_and_cleans(self):
        world,rc,report,csv_ok=self.run_mock('straight',fail_after=50)
        self.assertEqual(rc,2)
        self.assertTrue(report['cleanup_ok'])
        self.assertEqual(report['status'],'aborted')
        self.assertTrue(world.actor.destroyed)
        self.assertFalse(world.settings.synchronous_mode)


if __name__=='__main__':unittest.main(verbosity=2)
