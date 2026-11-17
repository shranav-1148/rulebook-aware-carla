"""Unit tests for pure parts of CARLA no-drive drivetrain diagnostics."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch

import carla_drivetrain_probe_m6b3 as m


class Tests(unittest.TestCase):
    def test_physics_fields(self):
        p=S(mass=1845,max_rpm=6000,use_gear_autobox=True,gear_switch_time=.5,
            clutch_strength=10,final_ratio=3.,reverse_gear_ratio=-3.,drag_coefficient=.3,
            torque_curve=[S(x=0,y=400),S(x=1000,y=700)],
            forward_gears=[S(ratio=2.0,down_ratio=0.5,up_ratio=.7)],
            wheels=[S(radius=35,max_steer_angle=70,tire_friction=2.,position=S(x=1,y=2,z=-3))])
        r=m.physics_report(p)
        self.assertTrue(r['use_gear_autobox'])
        self.assertEqual(r['forward_gears'][0]['ratio'],2.)
        self.assertEqual(r['wheels'][0]['position'],[1.,2.,-3.])
        self.assertEqual(r['torque_curve_rpm_torque'][1],[1000.,700.])
    def test_missing_optional_fields_are_safe(self):
        r=m.physics_report(S())
        self.assertIsNone(r['use_gear_autobox'])
        self.assertEqual(r['forward_gears'],[])
        self.assertEqual(r['wheels'],[])
    def test_nonfinite_values_not_serialized(self):
        self.assertIsNone(m.number_or_none(float('nan')))
        self.assertIsNone(m.number_or_none(float('inf')))
        self.assertIsNone(m.number_or_none('bad'))
    def test_control_report_does_not_infer_gear_engagement(self):
        c=S(throttle=.25,brake=0,hand_brake=False,reverse=False,manual_gear_shift=False,gear=0)
        r=m.control_report(c)
        self.assertEqual(r['gear_field'],0)
        self.assertFalse(r['manual_gear_shift'])
    def test_successful_probe_does_not_drive_and_cleans_actor(self):
        class Vehicle:
            id=12
            type_id='vehicle.tesla.model3'
            def __init__(self): self.destroyed=False
            def get_physics_control(self):return S(use_gear_autobox=True,forward_gears=[S(ratio=1.)],torque_curve=[])
            def get_control(self):return S(throttle=0,brake=0,hand_brake=False,reverse=False,manual_gear_shift=False,gear=0)
            def get_failure_state(self):return 'NONE'
            def destroy(self):self.destroyed=True;return True
        vehicle=Vehicle()
        class World:
            def get_settings(self):return S(synchronous_mode=False)
            def try_spawn_actor(self,*a):return vehicle
            def tick(self):raise AssertionError('Probe must never tick')
            def apply_settings(self,*a):raise AssertionError('Probe must not change settings')
        world=World()
        carla=S(Transform=lambda loc,rot:S(location=loc,rotation=rot),
                Location=lambda x,y,z:S(x=x,y=y,z=z),
                Rotation=lambda yaw:S(yaw=yaw))
        args=S(out=None,observation='not-used')
        obs={'body_center_cr_xy_m':[384.591,1.993], 'yaw_cr_rad':3.14159,'source_frame':123}
        with tempfile.TemporaryDirectory() as d:
            args.out=Path(d)
            with (patch.object(m,'read_stable_observation',return_value=obs),
                  patch.object(m,'preflight',return_value=(carla,None,world,S(id='vehicle.tesla.model3'),S(location=S(z=0)),{}))):
                self.assertEqual(m.run(args),0)
            self.assertTrue(vehicle.destroyed)
            report=json.loads((Path(d)/'drivetrain_report.json').read_text())
            self.assertTrue(report['no_client_ticks_issued'])
            self.assertTrue(report['no_vehicle_control_applied'])
            self.assertTrue(report['cleanup']['actor_destroy_returned_success'])

    def test_probe_prevents_overwrite_before_spawning(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,'drivetrain_report.json').write_text('{}')
            with patch.object(m,'read_stable_observation',side_effect=AssertionError('should not get here')):
                with self.assertRaises(FileExistsError):m.run(S(out=Path(d),observation='not-used'))

    def test_write_report_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=m.write_report(d,{'status':'physics_read'})
            self.assertEqual(json.loads(p.read_text())['status'],'physics_read')
            with self.assertRaises(FileExistsError):
                m.write_report(d,{'status':'physics_read'})

if __name__=='__main__':
    unittest.main()
