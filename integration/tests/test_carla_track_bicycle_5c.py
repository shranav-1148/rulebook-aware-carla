"""Offline tests for Milestone 5C. No CARLA server or CARLA Python package required."""
import csv
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import carla_track_bicycle_5c as m


def samples():
    return [m.Reference(0.,384.6,2.0,math.pi,6.,0.,385.95,2.,0.,0.),
            m.Reference(.1,384.0,2.0,math.pi,6.,0.,385.35,2.,.6,0.),
            m.Reference(.2,383.4,2.1,math.pi-.04,6.,-2.,384.75,2.046,.2,.05)]


class TestTracking(unittest.TestCase):
    def test_xy_mirror_involution(self):
        x,y=m.to_carla_xy(384.6,2.)
        self.assertEqual((x,y),(384.6,-2.))
        self.assertEqual(m.to_carla_xy(x,y),(384.6,2.))

    def test_yaw_mirror(self):
        self.assertAlmostEqual(m.to_carla_yaw(math.pi/2),-math.pi/2)
        self.assertAlmostEqual(m.to_commonroad_yaw(m.to_carla_yaw(math.pi/3)),math.pi/3)

    def test_yaw_wrap_interpolation(self):
        a=m.Reference(0.,0.,0.,math.radians(179),6.,0.,0.,0.,0.,0.)
        b=m.Reference(1.,-1.,0.,math.radians(-179),6.,0.,-1.,0.,1.,0.)
        mid=m.interpolate(a,b,.5)
        self.assertAlmostEqual(abs(math.degrees(mid.yaw)),180,places=5)

    def test_time_interpolation(self):
        mid=m.at_time(samples(),.05)
        self.assertAlmostEqual(mid.x,384.3)
        self.assertAlmostEqual(mid.v,6.)

    def test_body_actor_offset_inverses(self):
        yaw=math.radians(-40.)
        body=(23.,-5.)
        box=(.3,-.08)
        origin=m.actor_origin_from_body_center(*body,yaw,box)
        recovered=m.body_center_from_actor_origin(*origin,yaw,box)
        self.assertAlmostEqual(recovered[0],body[0])
        self.assertAlmostEqual(recovered[1],body[1])

    def test_straight_pure_pursuit(self):
        self.assertAlmostEqual(m.pure_pursuit_angle(0,0,0,4,0,2.7),0.)

    def test_right_steer_positive_in_carla(self):
        self.assertGreater(m.pure_pursuit_angle(0,0,0,4,1,2.7),0.)
        self.assertLess(m.pure_pursuit_angle(0,0,0,4,-1,2.7),0.)

    def test_bicycle_straight_cr_westbound(self):
        a,b=samples()[:2]
        carla_yaw=m.to_carla_yaw(a.yaw)
        self.assertAlmostEqual(math.cos(carla_yaw),-1)
        self.assertAlmostEqual(b.x-a.x,-.6)

    def test_tracking_error_at_reference(self):
        r=samples()[0]
        e=m.tracking_errors(r,r.x,r.y,r.yaw)
        self.assertEqual(e['position_error_m'],0.)
        self.assertEqual(e['cross_track_error_m'],0.)

    def test_tracking_error_cross_track(self):
        r=m.Reference(0,0,0,0,6,0,0,0,0,0)
        e=m.tracking_errors(r,0,2,0)
        self.assertAlmostEqual(e['cross_track_error_m'],2.)

    def test_lookahead_follows_sequence(self):
        target=m.target_at_lookahead(samples(),385.95,-2.,.7)
        self.assertAlmostEqual(target.t,.2)

    def test_steering_curve_interpolation(self):
        class Point:
            def __init__(self,x,y): self.x=x;self.y=y
        curve=[Point(0,1.0),Point(50,.5)]
        self.assertAlmostEqual(m.steering_curve_fraction(curve,25),.75)

    def test_controller_commands_bounded(self):
        refs=samples()
        steer,throttle,brake,_=m.command_for_state(
            refs,0,385.95,-2.,-math.pi,5.8,2.7,60,[],0.,.05,2.)
        self.assertLessEqual(abs(steer),.8)
        self.assertTrue(0<=throttle<=.8)
        self.assertTrue(0<=brake<=.6)

    def test_summary_metric(self):
        rows=[{'t_s':0.,'position_error_m':1.,'cross_track_error_m':0.5,
               'heading_error_deg':2.,'actual_speed_mps':6.},
              {'t_s':.1,'position_error_m':2.,'cross_track_error_m':1.,
               'heading_error_deg':3.,'actual_speed_mps':7.}]
        s=m.summarize_rows(rows,[],'completed')
        self.assertAlmostEqual(s['position_rmse_m'],math.sqrt(2.5))
        self.assertEqual(s['final_position_error_m'],2.)

    def test_csv_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'traj.csv'
            fields=['t_s','x_m','y_m','yaw_rad','speed_mps','steering_deg',
                    'rear_axle_x_m','rear_axle_y_m','station_m','offset_m']
            with p.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields)
                w.writeheader()
                for r in samples():
                    w.writerow(dict(zip(fields,[r.t,r.x,r.y,r.yaw,r.v,r.steering_deg,r.rear_x,r.rear_y,r.station,r.offset])))
            got=m.load_references(p)
            self.assertEqual(len(got),3)
            self.assertAlmostEqual(got[-1].y,2.1)
            # Offline default must not import CARLA or access a server.
            m.main(['--trajectory',str(p)])

    def test_csv_rejects_nonmonotonic(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'traj.csv'
            fields=['t_s','x_m','y_m','yaw_rad','speed_mps','steering_deg',
                    'rear_axle_x_m','rear_axle_y_m','station_m','offset_m']
            with p.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
                for r in [samples()[0],samples()[0]]:
                    w.writerow(dict(zip(fields,[r.t,r.x,r.y,r.yaw,r.v,r.steering_deg,r.rear_x,r.rear_y,r.station,r.offset])))
            with self.assertRaises(ValueError): m.load_references(p)

    def test_real_m4eb_csv_format_smoke(self):
        p=Path(__file__).resolve().parent/'output/town01_bicycle_graph_beam800_min40/safety_first_bicycle_samples.csv'
        if p.is_file():
            ref=m.load_references(p)
            self.assertGreater(len(ref),10)
            self.assertAlmostEqual(ref[0].t,0.)
            self.assertGreaterEqual(ref[-1].t,5.)


class TestControlLoopStub(unittest.TestCase):
    """Mocks the CARLA boundary to exercise cleanup and logging, not vehicle physics."""
    def test_straight_run_uses_vehicle_control_and_restores_settings(self):
        from types import SimpleNamespace as NS
        from unittest.mock import patch
        from copy import copy
        class Location:
            def __init__(self,x=0.,y=0.,z=0.):self.x=x;self.y=y;self.z=z
        class Rotation:
            def __init__(self,yaw=0.):self.yaw=yaw
        class Transform:
            def __init__(self,location=None,rotation=None):
                self.location=location or Location();self.rotation=rotation or Rotation()
        class Vector3D:
            def __init__(self,x,y,z):self.x=x;self.y=y;self.z=z
        class Control:
            def __init__(self,throttle=0.,steer=0.,brake=0.):
                self.throttle=throttle;self.steer=steer;self.brake=brake
        fake_carla=NS(Location=Location,Rotation=Rotation,Transform=Transform,
                      Vector3D=Vector3D,VehicleControl=Control)
        class FakeVehicle:
            def __init__(self,pose):
                self.pose=pose;self.velocity=Vector3D(0,0,0);self.bounding_box=NS(location=Location())
                self.id=7;self.controls=[];self.destroyed=False
            def set_autopilot(self,flag): pass
            def set_transform(self,pose):self.pose=pose
            def set_simulate_physics(self,flag): pass
            def set_target_velocity(self,v):self.velocity=v
            def get_physics_control(self):
                return NS(wheels=[NS(max_steer_angle=60),NS(max_steer_angle=60)],steering_curve=[])
            def apply_control(self,c):self.controls.append(c)
            def get_transform(self):return self.pose
            def get_velocity(self):return self.velocity
            def destroy(self):self.destroyed=True
        class FakeSensor:
            def __init__(self):self.destroyed=False
            def listen(self,fn):self.callback=fn
            def stop(self): pass
            def destroy(self):self.destroyed=True
        class FakeWorld:
            def __init__(self):
                self.settings=NS(synchronous_mode=False,fixed_delta_seconds=None)
                self.initial=self.get_settings();self.count=0;self.vehicle=None;self.sensor=None
            def get_settings(self):return copy(self.settings)
            def apply_settings(self,s):self.settings=copy(s)
            def try_spawn_actor(self,bp,t):
                self.vehicle=FakeVehicle(t);return self.vehicle
            def get_blueprint_library(self):return NS(find=lambda name:'sensor')
            def spawn_actor(self,bp,transform,attach_to=None):
                self.sensor=FakeSensor();return self.sensor
            def tick(self):
                self.count+=1
                v=self.vehicle
                if v is not None and v.controls:
                    # Control loop smoke test only: maintain initial velocity.
                    dt=self.settings.fixed_delta_seconds
                    v.pose.location.x+=v.velocity.x*dt
                    v.pose.location.y+=v.velocity.y*dt
                return self.count
            def get_snapshot(self):return NS(frame=self.count)
        class FakeBP:
            def has_attribute(self,name):return True
            def set_attribute(self,name,val):pass
        world=FakeWorld()
        refs=[m.Reference(t,384.6-6*t,2.,math.pi,6.,0.,385.95-6*t,2.,6*t,0.)
              for t in [i/10 for i in range(21)]]
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'mock_source.csv';source.write_text('test')
            args=NS(out=Path(tmp)/'out',trajectory=source,host='127.0.0.1',port=2000,
                    blueprint='vehicle.tesla.model3',fixed_dt=.05,lookahead=3.,
                    wheelbase=2.7,spawn_z_offset=.5,max_initial_error=1.5,
                    max_tracking_error=4.,max_tracking_heading_error=60.,max_speed=15.,
                    initialize_velocity=True, controller='position',
                    position_kp=0.6, max_speed_correction=2.0, max_commanded_speed=12.0)
            with patch.object(m,'server_preflight',return_value=(fake_carla,world,NS(location=Location(z=.05)),FakeBP(),NS(get_client_version=lambda:'0.9.16',get_server_version=lambda:'0.9.16'))):
                m.execute(args,refs)
            self.assertTrue((args.out/'summary.json').is_file())
            data=__import__('json').loads((args.out/'summary.json').read_text())
            self.assertEqual(data['status'],'completed')
            self.assertEqual(data['samples'],41)
            self.assertTrue(world.vehicle.destroyed)
            self.assertTrue(world.sensor.destroyed)
            self.assertFalse(world.settings.synchronous_mode)
            self.assertGreater(len(world.vehicle.controls),10)
            self.assertLess(data['max_position_error_m'],1.)
            self.assertEqual(data['controller_type'],'position')
            self.assertIn('along_track_rmse_m',data)
            self.assertTrue((args.out/'tracking_samples.csv').is_file())



class TestLongitudinalFeedback(unittest.TestCase):
    def make_ref(self, yaw=math.pi, speed=6.0):
        return m.Reference(0., 100., 2., yaw, speed, 0., 101.35, 2., 0., 0.)

    def test_westbound_behind_increases_target_speed(self):
        ref=self.make_ref()
        feedback=m.longitudinal_feedback(ref, 102., 2., 6.)
        self.assertAlmostEqual(feedback['along_lag_m'],2.)
        self.assertAlmostEqual(feedback['speed_correction_mps'],1.2)
        self.assertAlmostEqual(feedback['commanded_speed_mps'],7.2)
        self.assertGreater(feedback['throttle'],0.28)

    def test_westbound_ahead_reduces_target_speed(self):
        ref=self.make_ref()
        feedback=m.longitudinal_feedback(ref, 98., 2., 6.)
        self.assertAlmostEqual(feedback['along_lag_m'],-2.)
        self.assertAlmostEqual(feedback['commanded_speed_mps'],4.8)
        self.assertEqual(feedback['throttle'],0.0)
        self.assertGreater(feedback['brake'],0.0)

    def test_eastbound_lag_sign(self):
        ref=self.make_ref(yaw=0.)
        result=m.longitudinal_feedback(ref,98.,2.,6.)
        self.assertAlmostEqual(result['along_lag_m'],2.)

    def test_lateral_offset_does_not_cause_progress_correction(self):
        ref=self.make_ref(yaw=0.)
        result=m.longitudinal_feedback(ref,100.,20.,6.)
        self.assertAlmostEqual(result['along_lag_m'],0.)
        self.assertAlmostEqual(result['speed_correction_mps'],0.)

    def test_lag_saturation(self):
        ref=self.make_ref()
        result=m.longitudinal_feedback(ref,999.,2.,6.)
        self.assertEqual(result['speed_correction_mps'],2.)
        self.assertEqual(result['commanded_speed_mps'],8.)

    def test_ahead_saturation(self):
        ref=self.make_ref()
        result=m.longitudinal_feedback(ref,-500.,2.,6.)
        self.assertEqual(result['speed_correction_mps'],-2.)
        self.assertEqual(result['commanded_speed_mps'],4.)

    def test_target_speed_clamped_no_reverse(self):
        ref=self.make_ref(speed=0.0)
        result=m.longitudinal_feedback(ref,0.,2.,0.,max_speed_correction=3.)
        self.assertEqual(result['commanded_speed_mps'],0.)

    def test_target_speed_ceiling(self):
        ref=self.make_ref(speed=11.7)
        result=m.longitudinal_feedback(ref,110.,2.,7.,max_commanded_speed=12.)
        self.assertEqual(result['commanded_speed_mps'],12.)

    def test_zero_gain_matches_legacy_throttle_law(self):
        ref=self.make_ref()
        fb=m.longitudinal_feedback(ref,125.,2.,5.5,position_kp=0.)
        expected=m.clamp(0.28+0.23*(ref.v-5.5),0.,0.8)
        self.assertAlmostEqual(fb['throttle'],expected)
        self.assertEqual(fb['brake'],0.)

    def test_bounded_outputs(self):
        ref=self.make_ref(speed=9.)
        for pos in (-100.,-10.,100.,102.,1000.):
            for speed in (0.,2.,6.,10.,14.):
                fb=m.longitudinal_feedback(ref,pos,2.,speed)
                self.assertTrue(0 <= fb['throttle'] <= .8)
                self.assertTrue(0 <= fb['brake'] <= .6)
                self.assertFalse(fb['brake'] and fb['throttle'])

    def test_rejects_bad_inputs_and_gains(self):
        ref=self.make_ref()
        with self.assertRaises(ValueError):m.longitudinal_feedback(ref,float('nan'),2.,6.)
        with self.assertRaises(ValueError):m.longitudinal_feedback(ref,100.,2.,-1.)
        with self.assertRaises(ValueError):m.longitudinal_feedback(ref,100.,2.,6.,position_kp=-1.)
        with self.assertRaises(ValueError):m.longitudinal_feedback(ref,100.,2.,6.,max_speed_correction=7.)

    def test_feedback_row_metadata(self):
        ref=self.make_ref()
        actual={'x_cr':101.,'y_cr':2.,'yaw_cr':ref.yaw,'speed_mps':5.,'x_carla':101.,'y_carla':-2.}
        fb=m.longitudinal_feedback(ref,actual['x_cr'],actual['y_cr'],actual['speed_mps'])
        row=m.make_log_row(.05,ref,actual,0.,fb['throttle'],fb['brake'],123,fb)
        self.assertAlmostEqual(row['along_track_error_m'],-1.)
        self.assertAlmostEqual(row['along_lag_m'],1.)
        self.assertIn('commanded_speed_mps',row)

    def test_csv_diagnostics_and_aggregate(self):
        ref=self.make_ref()
        rowlist=[]
        for t,x in [(0,100.),(.05,101.)]:
            actual={'x_cr':x,'y_cr':2.,'yaw_cr':ref.yaw,'speed_mps':6.,'x_carla':x,'y_carla':-2.}
            rowlist.append(m.make_log_row(t,ref,actual,0.,.28,0.,100))
        s=m.summarize_rows(rowlist,[],'completed')
        self.assertAlmostEqual(s['along_track_rmse_m'],math.sqrt(.5))
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'track.csv';m.save_rows(f,rowlist)
            with f.open(newline='') as src:
                rows=list(csv.DictReader(src))
            self.assertEqual(len(rows),2)
            self.assertIn('speed_correction_mps',rows[0])

    def test_refuse_overwrite_before_connecting_to_carla(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);traj=root/'traj.csv';out=root/'run';out.mkdir()
            (out/'summary.json').write_text('{}')
            fields=['t_s','x_m','y_m','yaw_rad','speed_mps','steering_deg',
                    'rear_axle_x_m','rear_axle_y_m','station_m','offset_m']
            with traj.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
                for ref in samples():
                    w.writerow(dict(zip(fields,[ref.t,ref.x,ref.y,ref.yaw,ref.v,ref.steering_deg,
                                               ref.rear_x,ref.rear_y,ref.station,ref.offset])))
            with patch.object(m,'server_preflight',side_effect=AssertionError('Must not connect')):
                with self.assertRaises(SystemExit) as exc:
                    m.main(['--trajectory',str(traj),'--out',str(out),'--execute'])
            self.assertEqual(exc.exception.code,2)

    def test_reject_cli_unsafe_gain_before_connecting(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);traj=root/'traj.csv'
            fields=['t_s','x_m','y_m','yaw_rad','speed_mps','steering_deg',
                    'rear_axle_x_m','rear_axle_y_m','station_m','offset_m']
            with traj.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
                for ref in samples():
                    w.writerow(dict(zip(fields,[ref.t,ref.x,ref.y,ref.yaw,ref.v,ref.steering_deg,
                                               ref.rear_x,ref.rear_y,ref.station,ref.offset])))
            with patch.object(m,'server_preflight',side_effect=AssertionError('Must not connect')):
                with self.assertRaises(SystemExit) as exc:
                    m.main(['--trajectory',str(traj),'--execute','--position-kp','4'])
            self.assertEqual(exc.exception.code,2)

    def test_default_preflight_does_not_access_carla(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'trajectory.csv'
            fields=['t_s','x_m','y_m','yaw_rad','speed_mps','steering_deg',
                    'rear_axle_x_m','rear_axle_y_m','station_m','offset_m']
            with p.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
                for ref in samples():
                    w.writerow(dict(zip(fields,[ref.t,ref.x,ref.y,ref.yaw,ref.v,ref.steering_deg,
                                               ref.rear_x,ref.rear_y,ref.station,ref.offset])))
            m.main(['--trajectory',str(p)])


if __name__=='__main__':
    unittest.main()
