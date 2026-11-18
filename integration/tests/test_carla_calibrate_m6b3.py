"""Offline/mock tests. They do NOT validate CARLA vehicle physics or actual control safety."""
# Import dependency: json supplies types or behavior used by this test module.
import json
# Import dependency: math supplies types or behavior used by this test module.
import math
# Import dependency: tempfile supplies types or behavior used by this test module.
import tempfile
# Import dependency: pathlib supplies types or behavior used by this test module.
from pathlib import Path
# Import dependency: types supplies types or behavior used by this test module.
from types import SimpleNamespace as NS
# Import dependency: unittest supplies types or behavior used by this test module.
import unittest
# Import dependency: unittest.mock supplies types or behavior used by this test module.
from unittest.mock import patch
# Import dependency: copy supplies types or behavior used by this test module.
from copy import copy

# Import dependency: carla_calibrate_m6b3 supplies types or behavior used by this test module.
import carla_calibrate_m6b3 as m


# Helper routine: obj; exposes a focused operation or synthetic fixture while keeping test intent readable.
def obj(**kwargs):
    return NS(**kwargs)


# Helper routine: stable json; exposes a focused operation or synthetic fixture while keeping test intent readable.
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


# Helper routine: default obs; exposes a focused operation or synthetic fixture while keeping test intent readable.
def default_obs():
    return {'body_center_cr_xy_m':[384.59083741,1.99333765],
            'yaw_cr_rad':3.1410602876,'source_frame':967030,'sha256':'fakehash'}


# Test class TestPure: groups related setup, helpers, and assertions behind one fixture lifecycle.
class TestPure(unittest.TestCase):
    # Test method: verifies finite nan without changing the behavior under test.
    def test_finite_nan(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(ValueError):m.finite('v',float('nan'))
    # Test method: verifies wrap without changing the behavior under test.
    def test_wrap(self):
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertAlmostEqual(abs(m.wrap(3*math.pi)),math.pi)
    # Test method: verifies body center inverse without changing the behavior under test.
    def test_body_center_inverse(self):
        b=obj(x=.3,y=-.02)
        # Repeated scenario: exercise the related states independently and retain each assertion result.
        for yaw in (-math.pi,0,math.pi/2,.3):
            x,y=m.actor_origin_for_center(12,-9,yaw,b)
            cx,cy=m.center_from_actor(x,y,yaw,b)
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertAlmostEqual(cx,12)
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertAlmostEqual(cy,-9)
    # Test method: verifies parse valid observation without changing the behavior under test.
    def test_parse_valid_observation(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'obs.json';p.write_text(json.dumps(stable_json()))
            data=m.read_stable_observation(p)
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertEqual(data['source_frame'],967030)
    # Test method: verifies reject unstable without changing the behavior under test.
    def test_reject_unstable(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with tempfile.TemporaryDirectory() as t:
            doc=stable_json();doc['stability']['latest']['accepted']=False
            p=Path(t)/'obs.json';p.write_text(json.dumps(doc))
            # Context/exception scope: install the temporary mock or exception guard needed only for this check.
            with self.assertRaises(ValueError):m.read_stable_observation(p)
    # Test method: verifies reject wrong town without changing the behavior under test.
    def test_reject_wrong_town(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with tempfile.TemporaryDirectory() as t:
            doc=stable_json();doc['server']['map']='Carla/Maps/Town02'
            p=Path(t)/'obs.json';p.write_text(json.dumps(doc))
            # Context/exception scope: install the temporary mock or exception guard needed only for this check.
            with self.assertRaises(ValueError):m.read_stable_observation(p)
    # Test method: verifies reject mismatched start without changing the behavior under test.
    def test_reject_mismatched_start(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with tempfile.TemporaryDirectory() as t:
            doc=stable_json();doc['ego']['body_center_cr_xy_m'][0]=100
            p=Path(t)/'obs.json';p.write_text(json.dumps(doc))
            # Context/exception scope: install the temporary mock or exception guard needed only for this check.
            with self.assertRaises(ValueError):m.read_stable_observation(p)
    # Test method: verifies reject cleanup failed without changing the behavior under test.
    def test_reject_cleanup_failed(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with tempfile.TemporaryDirectory() as t:
            doc=stable_json();doc['cleanup_ok']=False
            p=Path(t)/'obs.json';p.write_text(json.dumps(doc))
            # Context/exception scope: install the temporary mock or exception guard needed only for this check.
            with self.assertRaises(ValueError):m.read_stable_observation(p)
    # Test method: verifies reject unsafe stable speed without changing the behavior under test.
    def test_reject_unsafe_stable_speed(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with tempfile.TemporaryDirectory() as t:
            doc=stable_json();doc['ego']['longitudinal_speed_mps']=-.3
            p=Path(t)/'obs.json';p.write_text(json.dumps(doc))
            # Context/exception scope: install the temporary mock or exception guard needed only for this check.
            with self.assertRaises(ValueError):m.read_stable_observation(p)
    # Test method: verifies motion gate zero is failure without changing the behavior under test.
    def test_motion_gate_zero_is_failure(self):
        row={'phase':'throttle','forward_speed_mps':0.,'distance_from_start_m':0.}
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(m.motion_during_throttle([row])['passed'])

    # Test method: verifies motion gate real response passes without changing the behavior under test.
    def test_motion_gate_real_response_passes(self):
        row={'phase':'throttle','forward_speed_mps':.5,'distance_from_start_m':.15}
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(m.motion_during_throttle([row])['passed'])

    # Test method: verifies readback control flags without changing the behavior under test.
    def test_readback_control_flags(self):
        c=obj(throttle=.25,steer=0.,brake=0.,hand_brake=False,reverse=False,manual_gear_shift=False,gear=1)
        r=m.applied_control_readback(obj(get_control=lambda: c))
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(r['applied_gear'],1)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(r['applied_hand_brake'])

    # Test method: verifies straight command phases without changing the behavior under test.
    def test_straight_command_phases(self):
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(m.plan_command('straight',0)[0],'throttle')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(m.plan_command('straight',1.5)[0],'coast')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(m.plan_command('straight',2.0)[0],'brake')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertLessEqual(m.plan_command('straight',0)[1],.25)
    # Test method: verifies manual first uses identical throttle profile without changing the behavior under test.
    def test_manual_first_uses_identical_throttle_profile(self):
        # Repeated scenario: exercise the related states independently and retain each assertion result.
        for t in (0, .5, 1.45, 1.5, 2, 4):
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertEqual(m.plan_command('manual_first',t),m.plan_command('straight',t))
    # Test method: verifies manual first control uses gear one without more throttle without changing the behavior under test.
    def test_manual_first_control_uses_gear_one_without_more_throttle(self):
        auto=m.commanded_vehicle_control(obj(VehicleControl=MockWorld.Control),'straight',.25,0.,0.)
        manual=m.commanded_vehicle_control(obj(VehicleControl=MockWorld.Control),'manual_first',.25,0.,0.)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(auto.manual_gear_shift)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(auto.gear,0)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(manual.manual_gear_shift)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(manual.gear,1)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(manual.throttle,auto.throttle)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(manual.hand_brake)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(manual.reverse)
    # Test method: verifies manual first requires explicit drive confirmation without changing the behavior under test.
    def test_manual_first_requires_explicit_drive_confirmation(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(SystemExit) as e:m.main(['--execute-manual-first'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(e.exception.code,2)
    # Test method: verifies manual first not accepted as automatic steering baseline without changing the behavior under test.
    def test_manual_first_not_accepted_as_automatic_steering_baseline(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'manual.json';p.write_text(json.dumps(dict(mode='manual_first',status='completed',
                     cleanup_ok=True,settings_restored=True,collision_count=0,
                     observation_sha256='fakehash',max_displacement_m=1.,max_total_speed_mps=1.,
                     throttle_motion_gate={'passed':True})))
            # Context/exception scope: install the temporary mock or exception guard needed only for this check.
            with self.assertRaises(ValueError):m.validate_baseline(p,'fakehash')
    # Test method: verifies wheelbase geometry world centimeters without changing the behavior under test.
    def test_wheelbase_geometry_world_centimeters(self):
        p=obj(wheels=[obj(position=obj(x=38297.2,y=-116.0)),
                       obj(position=obj(x=38297.3,y=-282.7)),
                       obj(position=obj(x=38597.7,y=-115.9)),
                       obj(position=obj(x=38597.8,y=-282.6))])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertAlmostEqual(m.estimated_wheel_geometry(p)['wheelbase_from_xy_m'],3.005,places=2)
    # Test method: verifies steering command bounded without changing the behavior under test.
    def test_steering_command_bounded(self):
        # Repeated scenario: exercise the related states independently and retain each assertion result.
        for sign in (-1,1):
            c=m.plan_command('steering',1.2,sign)
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertEqual(c[0],'steer_pulse')
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertAlmostEqual(c[2],sign*.08)

    # Test method: verifies manual steering reuses bounded original pulse without changing the behavior under test.
    def test_manual_steering_reuses_bounded_original_pulse(self):
        # Repeated scenario: exercise the related states independently and retain each assertion result.
        for t in (0., .8,1.15,1.5,1.9,2.5,4.):
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertEqual(m.plan_command('manual_steering',t),m.plan_command('steering',t))
    # Test method: verifies manual steering explicit first gear without changing the behavior under test.
    def test_manual_steering_explicit_first_gear(self):
        c=m.commanded_vehicle_control(obj(VehicleControl=MockWorld.Control),'manual_steering',.2,.08,0.)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(c.manual_gear_shift)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(c.gear,1)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(c.steer,.08)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(c.throttle,.2)
    # Test method: verifies manual steering requires confirm and baseline without changing the behavior under test.
    def test_manual_steering_requires_confirm_and_baseline(self):
        # Repeated scenario: exercise the related states independently and retain each assertion result.
        for argv in (['--execute-manual-steering'],['--execute-manual-steering','--confirm-drive']):
            # Context/exception scope: install the temporary mock or exception guard needed only for this check.
            with self.assertRaises(SystemExit) as e:m.main(argv)
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertEqual(e.exception.code,2)
    # Test method: verifies manual steering baseline accepts manual first not auto without changing the behavior under test.
    def test_manual_steering_baseline_accepts_manual_first_not_auto(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'s.json'
            doc=dict(mode='manual_first',status='completed',cleanup_ok=True,settings_restored=True,
                 collision_count=0,observation_sha256='fakehash',max_displacement_m=1.838,
                 max_total_speed_mps=1.614,throttle_motion_gate={'passed':True})
            path.write_text(json.dumps(doc))
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertEqual(m.validate_baseline(path,'fakehash',expected_mode='manual_first')['status'],'completed')
            # Context/exception scope: install the temporary mock or exception guard needed only for this check.
            with self.assertRaises(ValueError):m.validate_baseline(path,'fakehash',expected_mode='straight')
    # Test method: verifies manual steering baseline rejects bad cleanup without changing the behavior under test.
    def test_manual_steering_baseline_rejects_bad_cleanup(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'s.json'
            p.write_text(json.dumps(dict(mode='manual_first',status='completed',cleanup_ok=False,settings_restored=True,
                 collision_count=0,observation_sha256='fakehash',max_displacement_m=1.,max_total_speed_mps=1.,
                 throttle_motion_gate={'passed':True})))
            # Context/exception scope: install the temporary mock or exception guard needed only for this check.
            with self.assertRaises(ValueError):m.validate_baseline(p,'fakehash',expected_mode='manual_first')
    # Test method: verifies manual steering readback gate without changing the behavior under test.
    def test_manual_steering_readback_gate(self):
        row=dict(applied_throttle=.2,applied_steer=.08,applied_brake=0.,applied_hand_brake=False,
                 applied_reverse=False,applied_manual_gear_shift=True,applied_gear=1)
        m.manual_steering_control_check(row,'steer_pulse',.2,.08,0.)
        # Repeated scenario: exercise the related states independently and retain each assertion result.
        for change in ({'applied_steer':0.},{'applied_gear':0},{'applied_manual_gear_shift':False},
                       {'applied_brake':.5},{'applied_hand_brake':True}):
            wrong=dict(row,**change)
            # Context/exception scope: install the temporary mock or exception guard needed only for this check.
            with self.assertRaises(RuntimeError):m.manual_steering_control_check(wrong,'steer_pulse',.2,.08,0.)
    # Test method: verifies manual steering gate requires yaw and wheel without changing the behavior under test.
    def test_manual_steering_gate_requires_yaw_and_wheel(self):
        row=dict(phase='steer_pulse',forward_speed_mps=1.,wheel_front_left_deg=1.6,
                 wheel_front_right_deg=1.5,heading_from_start_deg=.2,applied_steer=.08)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(m.steering_pulse_gate([row])['passed'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(m.steering_pulse_gate([dict(row,heading_from_start_deg=0.)])['passed'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(m.steering_pulse_gate([dict(row,wheel_front_left_deg=0.,wheel_front_right_deg=0.)])['passed'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(m.steering_pulse_gate([dict(row,forward_speed_mps=0.)])['passed'])

    # Test method: verifies unknown command rejected without changing the behavior under test.
    def test_unknown_command_rejected(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(ValueError):m.plan_command('nope',0)
    # Test method: verifies enrich cross track without changing the behavior under test.
    def test_enrich_cross_track(self):
        start={'x_cr_m':0.,'y_cr_m':0.,'yaw_cr_rad':0.}
        row={'x_cr_m':2.,'y_cr_m':.2,'yaw_cr_rad':0.}
        r=m.enrich(row,start,('test',0.,0.,0.),1.)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertAlmostEqual(r['along_track_m'],2.)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertAlmostEqual(r['cross_track_m'],.2)
    # Test method: verifies speed limit fails without changing the behavior under test.
    def test_speed_limit_fails(self):
        row={'total_speed_mps':3.6,'forward_speed_mps':3.5,'distance_from_start_m':1.,
             'cross_track_m':0.,'heading_from_start_deg':0.,'vertical_speed_mps':0.}
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertIn('speed/reverse bound',m.evaluate_limits(row))
    # Test method: verifies displacement limit fails without changing the behavior under test.
    def test_displacement_limit_fails(self):
        row={'total_speed_mps':1.,'forward_speed_mps':1.,'distance_from_start_m':8.1,
             'cross_track_m':0.,'heading_from_start_deg':0.,'vertical_speed_mps':0.}
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertIn('displacement bound',m.evaluate_limits(row))
    # Test method: verifies kinematic sign hypothesis without changing the behavior under test.
    def test_kinematic_sign_hypothesis(self):
        r=[]
        # Repeated scenario: exercise the related states independently and retain each assertion result.
        for j in range(3):
            r.append({'t_s':j*.1,'x_cr_m':j*.1,'y_cr_m':0.,'yaw_cr_rad':0.,
                 'forward_speed_mps':1.,'wheel_front_left_deg':0.,'wheel_front_right_deg':0.})
        res=m.kinematic_model_comparison(r)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertAlmostEqual(res['sign_hypothesis_rmse_m']['+'],0.)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertAlmostEqual(res['sign_hypothesis_rmse_m']['-'],0.)
    # Test method: verifies baseline reject wrong source without changing the behavior under test.
    def test_baseline_reject_wrong_source(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'s.json'; p.write_text(json.dumps(dict(mode='straight',status='completed',
                  cleanup_ok=True,settings_restored=True,collision_count=0,
                  observation_sha256='other',max_displacement_m=1.,max_total_speed_mps=1.)))
            # Context/exception scope: install the temporary mock or exception guard needed only for this check.
            with self.assertRaises(ValueError):m.validate_baseline(p,'fakehash')
    # Test method: verifies legacy completed but stuck baseline rejected without changing the behavior under test.
    def test_legacy_completed_but_stuck_baseline_rejected(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'old.json'
            p.write_text(json.dumps(dict(mode='straight',status='completed',
                cleanup_ok=True,settings_restored=True,collision_count=0,
                observation_sha256='fakehash',max_displacement_m=2.15e-6,
                max_total_speed_mps=.00179)))
            # Context/exception scope: install the temporary mock or exception guard needed only for this check.
            with self.assertRaises(ValueError):m.validate_baseline(p,'fakehash')

    # Test method: verifies baseline accept valid without changing the behavior under test.
    def test_baseline_accept_valid(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'s.json'; p.write_text(json.dumps(dict(mode='straight',status='completed',
                  cleanup_ok=True,settings_restored=True,collision_count=0,
                  observation_sha256='fakehash',max_displacement_m=1.,max_total_speed_mps=1.,
                  throttle_motion_gate={'passed':True})))
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertEqual(m.validate_baseline(p,'fakehash')['status'],'completed')
    # Test method: verifies save never overwrite without changing the behavior under test.
    def test_save_never_overwrite(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'out'
            m.save_results(p,[],{'status':'test'})
            # Context/exception scope: install the temporary mock or exception guard needed only for this check.
            with self.assertRaises(FileExistsError):m.save_results(p,[],{'status':'test'})
    # Test method: verifies no drive by default without changing the behavior under test.
    def test_no_drive_by_default(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'obs.json';p.write_text(json.dumps(stable_json()))
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertEqual(m.main(['--observation',str(p)]),0)
    # Test method: verifies confirm without drive rejected without changing the behavior under test.
    def test_confirm_without_drive_rejected(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(SystemExit) as e:m.main(['--confirm-drive'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(e.exception.code,2)
    # Test method: verifies drive without confirm rejected without changing the behavior under test.
    def test_drive_without_confirm_rejected(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(SystemExit) as e:m.main(['--execute-straight'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(e.exception.code,2)
    # Test method: verifies steering without baseline rejected without changing the behavior under test.
    def test_steering_without_baseline_rejected(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(SystemExit) as e:m.main(['--execute-steering','--confirm-drive'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(e.exception.code,2)


# Test class TestPreflight: groups related setup, helpers, and assertions behind one fixture lifecycle.
class TestPreflight(unittest.TestCase):
    # Helper routine: stub; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def stub(self, vehicles=0, walkers=0, props=0, sync=False, lane_distance=0.,
             mesh_positions=(), unexpected_props=(), broken_mesh=False):
        # Test class FakeProp: groups related setup, helpers, and assertions behind one fixture lifecycle.
        class FakeProp:
            # Helper routine: init; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def __init__(self, ident, typ, x, y, broken=False):
                self.id,self.type_id=ident,typ
                self.x,self.y,self.broken=x,y,broken
            # Helper routine: get transform; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def get_transform(self):return obj(location=obj(x=self.x,y=self.y,z=0.))
            @property
            # Helper routine: bounding box; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def bounding_box(self):
                # Conditional branch: distinguish the expected state or exceptional edge before continuing validation.
                if self.broken:raise RuntimeError('mock unavailable geometry')
                pos=self.get_transform().location
                return obj(get_world_vertices=lambda pose:[obj(x=pos.x+sx,y=pos.y+sy,z=sz)
                       # Repeated scenario: exercise the related states independently and retain each assertion result.
                       for sx in (-.2,.2) for sy in (-.2,.2) for sz in (0.,.4)])
        actual_props = ([object() for _ in range(props)] +
                        [FakeProp(i,'static.prop.mesh',x,y) for i,(x,y) in enumerate(mesh_positions,100)] +
                        [FakeProp(i,'static.prop.constructioncone',x,y) for i,(x,y) in enumerate(unexpected_props,200)] +
                        ([FakeProp(300,'static.prop.mesh',0.,0.,True)] if broken_mesh else []))
        # Test class Actors: groups related setup, helpers, and assertions behind one fixture lifecycle.
        class Actors:
            # Helper routine: filter; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def filter(self, pattern):
                # Conditional branch: distinguish the expected state or exceptional edge before continuing validation.
                if pattern.startswith('vehicle'):return [object() for _ in range(vehicles)]
                # Conditional branch: distinguish the expected state or exceptional edge before continuing validation.
                if pattern.startswith('walker'):return [object() for _ in range(walkers)]
                return actual_props
        # Test class BP: groups related setup, helpers, and assertions behind one fixture lifecycle.
        class BP:
            id = 'vehicle.tesla.model3'
        # Test class Map: groups related setup, helpers, and assertions behind one fixture lifecycle.
        class Map:
            name = 'Carla/Maps/Town01'
            # Helper routine: get waypoint; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def get_waypoint(self, location, project_to_road=True, lane_type=None):
                return obj(transform=obj(location=obj(x=location.x,y=location.y+lane_distance,z=0.),
                                         rotation=obj(yaw=-180.)))
        # Test class World: groups related setup, helpers, and assertions behind one fixture lifecycle.
        class World:
            # Helper routine: get map; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def get_map(self):return Map()
            # Helper routine: get blueprint library; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def get_blueprint_library(self):return obj(find=lambda k:BP())
            # Helper routine: get actors; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def get_actors(self):return Actors()
            # Helper routine: get settings; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def get_settings(self):return obj(synchronous_mode=sync)
        # Test class Client: groups related setup, helpers, and assertions behind one fixture lifecycle.
        class Client:
            # Helper routine: init; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def __init__(self,*a):pass
            # Helper routine: set timeout; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def set_timeout(self,n):pass
            # Helper routine: get client version; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def get_client_version(self):return '0.9.16'
            # Helper routine: get server version; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def get_server_version(self):return '0.9.16'
            # Helper routine: get world; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def get_world(self):return World()
        fake=obj(Client=Client,Location=lambda x,y,z:obj(x=x,y=y,z=z),LaneType=obj(Driving='Driving'))
        return fake
    # Helper routine: run preflight; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def run_preflight(self,fake,drive=False):
        # Import dependency: sys supplies types or behavior used by this test module.
        import sys
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with patch.dict(sys.modules, {'carla':fake}):
            return m.preflight(obj(host='127.0.0.1',port=2000,timeout=10.),default_obs(),drive)
    # Test method: verifies read only inspection accepts empty async world without changing the behavior under test.
    def test_read_only_inspection_accepts_empty_async_world(self):
        ret=self.run_preflight(self.stub())
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(ret[-1]['existing_vehicles'],0)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(len(ret[-1]['ahead_checks']),4)
    # Test method: verifies drive refuses occupied world without changing the behavior under test.
    def test_drive_refuses_occupied_world(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(RuntimeError):self.run_preflight(self.stub(vehicles=1),drive=True)
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(RuntimeError):self.run_preflight(self.stub(walkers=1),drive=True)
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(RuntimeError):self.run_preflight(self.stub(props=1),drive=True)
    # Test method: verifies drive accepts 115 distant mesh props without changing the behavior under test.
    def test_drive_accepts_115_distant_mesh_props(self):
        far = [(0.,0.)]*115  # far from the validated x=384.59 CARLA start
        info=self.run_preflight(self.stub(mesh_positions=far),drive=True)[-1]
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(info['existing_props'],115)
        screen=info['static_prop_screen']
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(screen['clear_for_calibration'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(screen['permitted_distant_mesh_count'],115)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(screen['nearby_count'],0)

    # Test method: verifies drive rejects mesh intersecting corridor without changing the behavior under test.
    def test_drive_rejects_mesh_intersecting_corridor(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaisesRegex(RuntimeError,'nearby=1'):
            self.run_preflight(self.stub(mesh_positions=[(380.,-1.993)]),drive=True)

    # Test method: verifies drive rejects unexpected prop even if far without changing the behavior under test.
    def test_drive_rejects_unexpected_prop_even_if_far(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaisesRegex(RuntimeError,'unexpected_types=1'):
            self.run_preflight(self.stub(unexpected_props=[(0.,0.)]),drive=True)

    # Test method: verifies drive rejects uninspectable mesh without changing the behavior under test.
    def test_drive_rejects_uninspectable_mesh(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaisesRegex(RuntimeError,'uninspectable=1'):
            self.run_preflight(self.stub(broken_mesh=True),drive=True)

    # Test method: verifies inspect server reports nearby without driving without changing the behavior under test.
    def test_inspect_server_reports_nearby_without_driving(self):
        screen=self.run_preflight(self.stub(mesh_positions=[(380.,-1.993)]),drive=False)[-1]['static_prop_screen']
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(screen['clear_for_calibration'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(screen['nearby_count'],1)

    # Test method: verifies drive refuses already synchronous without changing the behavior under test.
    def test_drive_refuses_already_synchronous(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(RuntimeError):self.run_preflight(self.stub(sync=True),drive=True)
    # Test method: verifies preflight rejects misaligned lane without changing the behavior under test.
    def test_preflight_rejects_misaligned_lane(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(RuntimeError):self.run_preflight(self.stub(lane_distance=.6))


# Test class MockWorld: groups related setup, helpers, and assertions behind one fixture lifecycle.
class MockWorld:
    # Test class Location: groups related setup, helpers, and assertions behind one fixture lifecycle.
    class Location:
        # Helper routine: init; exposes a focused operation or synthetic fixture while keeping test intent readable.
        def __init__(self,x=0.,y=0.,z=0.):self.x,self.y,self.z=x,y,z
    # Test class Rotation: groups related setup, helpers, and assertions behind one fixture lifecycle.
    class Rotation:
        # Helper routine: init; exposes a focused operation or synthetic fixture while keeping test intent readable.
        def __init__(self,yaw=0.):self.yaw=yaw
    # Test class Transform: groups related setup, helpers, and assertions behind one fixture lifecycle.
    class Transform:
        # Helper routine: init; exposes a focused operation or synthetic fixture while keeping test intent readable.
        def __init__(self,location=None,rotation=None):
            self.location=location if location is not None else MockWorld.Location()
            self.rotation=rotation if rotation is not None else MockWorld.Rotation()
    # Test class Control: groups related setup, helpers, and assertions behind one fixture lifecycle.
    class Control:
        # Helper routine: init; exposes a focused operation or synthetic fixture while keeping test intent readable.
        def __init__(self,throttle=0.,steer=0.,brake=0.,hand_brake=False,reverse=False,manual_gear_shift=False,gear=0):
            self.throttle,self.steer,self.brake=throttle,steer,brake
            self.hand_brake,self.reverse,self.manual_gear_shift,self.gear=hand_brake,reverse,manual_gear_shift,gear
    # Helper routine: init; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def __init__(self,collision=False,fail_after=None,stuck=False,brakes_stuck=False,requires_first_gear=False,steering_stuck=False):
        self.settings=obj(synchronous_mode=False,fixed_delta_seconds=None)
        self.count=0;self.actor=None;self.sensor=None
        self.force_collision=collision;self.fail_after=fail_after
        self.stuck=stuck;self.brakes_stuck=brakes_stuck;self.requires_first_gear=requires_first_gear
        self.steering_stuck=steering_stuck
        self.carla=obj(Location=self.Location,Rotation=self.Rotation,Transform=self.Transform,
                       VehicleControl=self.Control,VehicleWheelLocation=obj(FL_Wheel=0,FR_Wheel=1))
    # Helper routine: get settings; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def get_settings(self):return copy(self.settings)
    # Helper routine: apply settings; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def apply_settings(self,s):self.settings=copy(s)
    # Helper routine: try spawn actor; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def try_spawn_actor(self,bp,pose):
        self.actor=MockVehicle(pose,stuck=self.stuck,brakes_stuck=self.brakes_stuck,requires_first_gear=self.requires_first_gear,steering_stuck=self.steering_stuck)
        return self.actor
    # Helper routine: get blueprint library; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def get_blueprint_library(self):return obj(find=lambda name:obj(id=name))
    # Helper routine: spawn actor; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def spawn_actor(self,bp,transform,attach_to=None):
        self.sensor=MockSensor()
        return self.sensor
    # Helper routine: tick; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def tick(self,timeout=10.):
        self.count+=1
        # Conditional branch: distinguish the expected state or exceptional edge before continuing validation.
        if self.fail_after is not None and self.count>=self.fail_after:
            raise RuntimeError('mock tick failure')
        # Conditional branch: distinguish the expected state or exceptional edge before continuing validation.
        if self.actor is not None:
            self.actor.update(self.settings.fixed_delta_seconds or .05)
            # Conditional branch: distinguish the expected state or exceptional edge before continuing validation.
            if self.force_collision and self.count==32 and self.sensor and self.sensor.listener:
                self.sensor.listener(obj(frame=self.count,other_actor=obj(id=5,type_id='static.prop.test')))
        return self.count
    # Helper routine: get snapshot; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def get_snapshot(self):
        actor=self.actor
        # Test class SnapActor: groups related setup, helpers, and assertions behind one fixture lifecycle.
        class SnapActor:
            # Helper routine: get transform; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def get_transform(self):return actor.pose
            # Helper routine: get velocity; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def get_velocity(self):return obj(x=actor.v*math.cos(math.radians(actor.pose.rotation.yaw)),
                                               y=actor.v*math.sin(math.radians(actor.pose.rotation.yaw)),z=0.)
            # Helper routine: get acceleration; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def get_acceleration(self):return obj(x=0.,y=0.,z=0.)
        return obj(frame=self.count,find=lambda id:SnapActor() if id==actor.id else None,
                   timestamp=obj(elapsed_seconds=self.count*.05))


# Test class MockVehicle: groups related setup, helpers, and assertions behind one fixture lifecycle.
class MockVehicle:
    # Helper routine: init; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def __init__(self,pose,stuck=False,brakes_stuck=False,requires_first_gear=False,steering_stuck=False):
        self.id=10;self.pose=pose;self.v=0.
        self.stuck=stuck;self.brakes_stuck=brakes_stuck;self.requires_first_gear=requires_first_gear
        self.steering_stuck=steering_stuck
        self.bounding_box=obj(location=obj(x=.029,y=0.,z=.736))
        self.control=MockWorld.Control();self.controls=[];self.destroyed=False
    # Helper routine: set transform; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def set_transform(self,t):self.pose=t
    # Helper routine: set autopilot; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def set_autopilot(self,x):pass
    # Helper routine: set simulate physics; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def set_simulate_physics(self,x):pass
    # Helper routine: apply control; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def apply_control(self,c):self.control=c;self.controls.append(c)
    # Helper routine: get control; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def get_control(self):
        # Conditional branch: distinguish the expected state or exceptional edge before continuing validation.
        if self.brakes_stuck and self.control.throttle>0:
            c=copy(self.control);c.brake=.8;return c
        return self.control
    # Helper routine: get wheel steer angle; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def get_wheel_steer_angle(self,idx):return 0. if self.steering_stuck else self.control.steer*20.
    # Helper routine: get physics control; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def get_physics_control(self):return obj(wheels=[obj(max_steer_angle=60.),obj(max_steer_angle=60.)],mass=2000.)
    # Helper routine: update; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def update(self,dt):
        # Conditional branch: distinguish the expected state or exceptional edge before continuing validation.
        if self.stuck:return
        c=self.control
        # Conditional branch: distinguish the expected state or exceptional edge before continuing validation.
        if self.requires_first_gear and not (c.manual_gear_shift and c.gear==1):return
        self.v=max(0.,self.v+dt*(3*c.throttle-5*c.brake-.02))
        yaw=math.radians(self.pose.rotation.yaw)
        yaw+=self.v*math.tan(math.radians(0. if self.steering_stuck else 20*c.steer))/2.7*dt
        self.pose.rotation.yaw=math.degrees(yaw)
        self.pose.location.x+=dt*self.v*math.cos(yaw)
        self.pose.location.y+=dt*self.v*math.sin(yaw)
    # Helper routine: destroy; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def destroy(self):self.destroyed=True;return True


# Test class MockSensor: groups related setup, helpers, and assertions behind one fixture lifecycle.
class MockSensor:
    # Helper routine: init; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def __init__(self):self.listener=None;self.destroyed=False
    # Helper routine: listen; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def listen(self,fn):self.listener=fn
    # Helper routine: stop; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def stop(self):pass
    # Helper routine: destroy; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def destroy(self):self.destroyed=True;return True


# Test class TestLifecycle: groups related setup, helpers, and assertions behind one fixture lifecycle.
class TestLifecycle(unittest.TestCase):
    # Helper routine: run mock; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def run_mock(self,mode,collision=False,fail_after=None,stuck=False,brakes_stuck=False,requires_first_gear=False,steering_stuck=False,steer_sign=1):
        world=MockWorld(collision=collision,fail_after=fail_after,stuck=stuck,brakes_stuck=brakes_stuck,requires_first_gear=requires_first_gear,steering_stuck=steering_stuck)
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with tempfile.TemporaryDirectory() as t:
            args=obj(out=Path(t)/'run',timeout=10.,steer_sign=steer_sign)
            pre=obj(location=obj(z=0.))
            # Context/exception scope: install the temporary mock or exception guard needed only for this check.
            with patch.object(m,'preflight',return_value=(world.carla,None,world,obj(id='vehicle.tesla.model3',
                 has_attribute=lambda k:False),pre,dict(map='Carla/Maps/Town01', static_prop_screen=dict(total=0, permitted_distant_mesh_count=0, nearby_count=0, unexpected_type_count=0, uninspectable_count=0, clear_for_calibration=True)))):
                rc=m.drive(args,default_obs(),mode)
            report=json.loads((args.out/'summary.json').read_text())
            csv_present=(args.out/'calibration_samples.csv').exists()
            return world,rc,report,csv_present
    # Test method: verifies straight finishes restores cleanup without changing the behavior under test.
    def test_straight_finishes_restores_cleanup(self):
        world,rc,report,csv_ok=self.run_mock('straight')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(rc,0)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(report['status'],'completed')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(report['cleanup_ok'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(report['settings_restored'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(world.settings.synchronous_mode)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(world.actor.destroyed)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(world.sensor.destroyed)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(csv_ok)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertGreater(report['max_displacement_m'],.5)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(report['rulebook_replanning_enabled'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(report['obstacle_spawned'])
    # Test method: verifies manual first executes without affecting auto baseline without changing the behavior under test.
    def test_manual_first_executes_without_affecting_auto_baseline(self):
        world,rc,report,csv_ok=self.run_mock('manual_first',requires_first_gear=True)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(rc,0)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(report['status'],'completed')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(report['mode'],'manual_first')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(report['cleanup_ok'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(report['settings_restored'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(report['configuration']['gear_diagnostic'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(report['applied_control_diagnostics']['first_throttle_readback']['applied_manual_gear_shift'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(report['applied_control_diagnostics']['observed_gears_during_throttle'],[1])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(report['rulebook_replanning_enabled'])
    # Test method: verifies auto control cannot move manual only mock vehicle without changing the behavior under test.
    def test_auto_control_cannot_move_manual_only_mock_vehicle(self):
        world,rc,report,csv_ok=self.run_mock('straight',requires_first_gear=True)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(rc,2)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(report['throttle_motion_gate']['passed'])
    # Test method: verifies manual first still fails stuck vehicle without changing the behavior under test.
    def test_manual_first_still_fails_stuck_vehicle(self):
        world,rc,report,csv_ok=self.run_mock('manual_first',stuck=True)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(rc,2)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertIn('No measurable acceleration',report['abort_reason'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(report['cleanup_ok'])
    # Test method: verifies steering finishes without changing the behavior under test.
    def test_steering_finishes(self):
        world,rc,report,csv_ok=self.run_mock('steering')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(rc,0)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(report['status'],'completed')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertGreater(report['max_abs_heading_change_deg'],0.)

    # Test method: verifies manual steering completes with measured response without changing the behavior under test.
    def test_manual_steering_completes_with_measured_response(self):
        world,rc,report,csv_ok=self.run_mock('manual_steering',requires_first_gear=True)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(rc,0)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(report['status'],'completed')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(report['throttle_motion_gate']['passed'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(report['steering_pulse_gate']['passed'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertGreater(report['steering_pulse_gate']['peak_abs_heading_change_deg'],.1)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(report['steering_diagnostics']['requested_steer_sign'],1)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(report['cleanup_ok'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(report['settings_restored'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(csv_ok)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(report['rulebook_replanning_enabled'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(report['obstacle_spawned'])
    # Test method: verifies manual steering negative sign produces opposite yaw without changing the behavior under test.
    def test_manual_steering_negative_sign_produces_opposite_yaw(self):
        _,r1,rep1,_=self.run_mock('manual_steering',requires_first_gear=True,steer_sign=1)
        _,r2,rep2,_=self.run_mock('manual_steering',requires_first_gear=True,steer_sign=-1)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual((r1,r2),(0,0))
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertLess(rep1['steering_diagnostics']['final_heading_change_deg']*rep2['steering_diagnostics']['final_heading_change_deg'],0)
    # Test method: verifies manual steering no gear cannot move without changing the behavior under test.
    def test_manual_steering_no_gear_cannot_move(self):
        world,rc,report,csv_ok=self.run_mock('manual_steering',requires_first_gear=True,stuck=True)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(rc,2)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertIn('No measurable manual-first acceleration',report['abort_reason'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(report['cleanup_ok'])
    # Test method: verifies manual steering no wheel turn aborts without changing the behavior under test.
    def test_manual_steering_no_wheel_turn_aborts(self):
        world,rc,report,csv_ok=self.run_mock('manual_steering',requires_first_gear=True,steering_stuck=True)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(rc,2)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertIn('No measurable steering response',report['abort_reason'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(report['steering_pulse_gate']['passed'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(report['settings_restored'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(report['cleanup_ok'])
    # Test method: verifies manual steering collision aborts without changing the behavior under test.
    def test_manual_steering_collision_aborts(self):
        world,rc,report,csv_ok=self.run_mock('manual_steering',requires_first_gear=True,collision=True)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(rc,2)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(report['status'],'aborted')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertGreater(report['collision_count'],0)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(world.actor.destroyed)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(world.settings.synchronous_mode)

    # Test method: verifies stuck vehicle fails not completed without changing the behavior under test.
    def test_stuck_vehicle_fails_not_completed(self):
        world,rc,report,csv_ok=self.run_mock('straight',stuck=True)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(rc,2)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(report['status'],'aborted')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertIn('No measurable acceleration',report['abort_reason'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(report['throttle_motion_gate']['passed'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(report['cleanup_ok'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(report['settings_restored'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(report['applied_control_diagnostics']['last_throttle_readback']['applied_throttle'],.25)

    # Test method: verifies control override is detected without changing the behavior under test.
    def test_control_override_is_detected(self):
        world,rc,report,csv_ok=self.run_mock('straight',brakes_stuck=True)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(rc,2)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertIn('does not match applied CARLA control',report['abort_reason'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(report['cleanup_ok'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(report['settings_restored'])

    # Test method: verifies collision aborts and cleans without changing the behavior under test.
    def test_collision_aborts_and_cleans(self):
        world,rc,report,csv_ok=self.run_mock('straight',collision=True)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(rc,2)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(report['status'],'aborted')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertGreater(report['collision_count'],0)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(report['cleanup_ok'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(world.actor.destroyed)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(world.settings.synchronous_mode)
    # Test method: verifies tick failure aborts and cleans without changing the behavior under test.
    def test_tick_failure_aborts_and_cleans(self):
        world,rc,report,csv_ok=self.run_mock('straight',fail_after=50)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(rc,2)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(report['cleanup_ok'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(report['status'],'aborted')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(world.actor.destroyed)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(world.settings.synchronous_mode)


# Conditional branch: distinguish the expected state or exceptional edge before continuing validation.
if __name__=='__main__':unittest.main(verbosity=2)
