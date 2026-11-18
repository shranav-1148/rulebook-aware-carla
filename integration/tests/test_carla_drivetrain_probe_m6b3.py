"""Unit tests for pure parts of CARLA no-drive drivetrain diagnostics."""
# Import dependency: json supplies types or behavior used by this test module.
import json
# Import dependency: tempfile supplies types or behavior used by this test module.
import tempfile
# Import dependency: unittest supplies types or behavior used by this test module.
import unittest
# Import dependency: pathlib supplies types or behavior used by this test module.
from pathlib import Path
# Import dependency: types supplies types or behavior used by this test module.
from types import SimpleNamespace as S
# Import dependency: unittest.mock supplies types or behavior used by this test module.
from unittest.mock import patch

# Import dependency: carla_drivetrain_probe_m6b3 supplies types or behavior used by this test module.
import carla_drivetrain_probe_m6b3 as m


# Test class Tests: groups related setup, helpers, and assertions behind one fixture lifecycle.
class Tests(unittest.TestCase):
    # Test method: verifies physics fields without changing the behavior under test.
    def test_physics_fields(self):
        p=S(mass=1845,max_rpm=6000,use_gear_autobox=True,gear_switch_time=.5,
            clutch_strength=10,final_ratio=3.,reverse_gear_ratio=-3.,drag_coefficient=.3,
            torque_curve=[S(x=0,y=400),S(x=1000,y=700)],
            forward_gears=[S(ratio=2.0,down_ratio=0.5,up_ratio=.7)],
            wheels=[S(radius=35,max_steer_angle=70,tire_friction=2.,position=S(x=1,y=2,z=-3))])
        r=m.physics_report(p)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(r['use_gear_autobox'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(r['forward_gears'][0]['ratio'],2.)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(r['wheels'][0]['position'],[1.,2.,-3.])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(r['torque_curve_rpm_torque'][1],[1000.,700.])
    # Test method: verifies missing optional fields are safe without changing the behavior under test.
    def test_missing_optional_fields_are_safe(self):
        r=m.physics_report(S())
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertIsNone(r['use_gear_autobox'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(r['forward_gears'],[])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(r['wheels'],[])
    # Test method: verifies nonfinite values not serialized without changing the behavior under test.
    def test_nonfinite_values_not_serialized(self):
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertIsNone(m.number_or_none(float('nan')))
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertIsNone(m.number_or_none(float('inf')))
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertIsNone(m.number_or_none('bad'))
    # Test method: verifies control report does not infer gear engagement without changing the behavior under test.
    def test_control_report_does_not_infer_gear_engagement(self):
        c=S(throttle=.25,brake=0,hand_brake=False,reverse=False,manual_gear_shift=False,gear=0)
        r=m.control_report(c)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(r['gear_field'],0)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(r['manual_gear_shift'])
    # Test method: verifies successful probe does not drive and cleans actor without changing the behavior under test.
    def test_successful_probe_does_not_drive_and_cleans_actor(self):
        # Test class Vehicle: groups related setup, helpers, and assertions behind one fixture lifecycle.
        class Vehicle:
            id=12
            type_id='vehicle.tesla.model3'
            # Helper routine: init; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def __init__(self): self.destroyed=False
            # Helper routine: get physics control; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def get_physics_control(self):return S(use_gear_autobox=True,forward_gears=[S(ratio=1.)],torque_curve=[])
            # Helper routine: get control; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def get_control(self):return S(throttle=0,brake=0,hand_brake=False,reverse=False,manual_gear_shift=False,gear=0)
            # Helper routine: get failure state; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def get_failure_state(self):return 'NONE'
            # Helper routine: destroy; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def destroy(self):self.destroyed=True;return True
        vehicle=Vehicle()
        # Test class World: groups related setup, helpers, and assertions behind one fixture lifecycle.
        class World:
            # Helper routine: get settings; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def get_settings(self):return S(synchronous_mode=False)
            # Helper routine: try spawn actor; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def try_spawn_actor(self,*a):return vehicle
            # Helper routine: tick; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def tick(self):raise AssertionError('Probe must never tick')
            # Helper routine: apply settings; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def apply_settings(self,*a):raise AssertionError('Probe must not change settings')
        world=World()
        carla=S(Transform=lambda loc,rot:S(location=loc,rotation=rot),
                Location=lambda x,y,z:S(x=x,y=y,z=z),
                Rotation=lambda yaw:S(yaw=yaw))
        args=S(out=None,observation='not-used')
        obs={'body_center_cr_xy_m':[384.591,1.993], 'yaw_cr_rad':3.14159,'source_frame':123}
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with tempfile.TemporaryDirectory() as d:
            args.out=Path(d)
            # Context/exception scope: install the temporary mock or exception guard needed only for this check.
            with (patch.object(m,'read_stable_observation',return_value=obs),
                  patch.object(m,'preflight',return_value=(carla,None,world,S(id='vehicle.tesla.model3'),S(location=S(z=0)),{}))):
                # Observable assertion: compare the resulting value or state with the contract documented by this test.
                self.assertEqual(m.run(args),0)
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertTrue(vehicle.destroyed)
            report=json.loads((Path(d)/'drivetrain_report.json').read_text())
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertTrue(report['no_client_ticks_issued'])
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertTrue(report['no_vehicle_control_applied'])
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertTrue(report['cleanup']['actor_destroy_returned_success'])

    # Test method: verifies probe prevents overwrite before spawning without changing the behavior under test.
    def test_probe_prevents_overwrite_before_spawning(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with tempfile.TemporaryDirectory() as d:
            Path(d,'drivetrain_report.json').write_text('{}')
            # Context/exception scope: install the temporary mock or exception guard needed only for this check.
            with patch.object(m,'read_stable_observation',side_effect=AssertionError('should not get here')):
                # Context/exception scope: install the temporary mock or exception guard needed only for this check.
                with self.assertRaises(FileExistsError):m.run(S(out=Path(d),observation='not-used'))

    # Test method: verifies write report and refuse overwrite without changing the behavior under test.
    def test_write_report_and_refuse_overwrite(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with tempfile.TemporaryDirectory() as d:
            p=m.write_report(d,{'status':'physics_read'})
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertEqual(json.loads(p.read_text())['status'],'physics_read')
            # Context/exception scope: install the temporary mock or exception guard needed only for this check.
            with self.assertRaises(FileExistsError):
                m.write_report(d,{'status':'physics_read'})

# Conditional branch: distinguish the expected state or exceptional edge before continuing validation.
if __name__=='__main__':
    unittest.main()
