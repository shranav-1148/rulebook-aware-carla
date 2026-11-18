"""M6B-5 new-route-reference offline/mocked tests. These are NOT live CARLA tests."""
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

# Import dependency: shapely.geometry supplies types or behavior used by this test module.
from shapely.geometry import box
# Import dependency: carla_route_reference_m6b5 supplies types or behavior used by this test module.
import carla_route_reference_m6b5 as ref
# Import dependency: carla_route_track_m6b5 supplies types or behavior used by this test module.
import carla_route_track_m6b5 as track
# Import dependency: carla_model_m6b4 supplies types or behavior used by this test module.
from carla_model_m6b4 import VERSION,digest
# Import dependency: test_carla_m6b4 supplies types or behavior used by this test module.
from test_carla_m6b4 import FakeWorld,FakeBlueprint,FakeCarla

X,Y=384.5906848222465,1.9933077714750864
YAW=3.1410605540039955


# Helper routine: geometry; exposes a focused operation or synthetic fixture while keeping test intent readable.
def geometry(curved=False, narrow=False):
    route=[(0.,X,Y),(100.,X-100,Y)]
    road=box(X-110,Y-.25 if narrow else Y-10.,X+10,Y+.25 if narrow else Y+10.)
    # Helper routine: pose; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def pose(_route,s):
        # Conditional branch: distinguish the expected state or exceptional edge before continuing validation.
        if curved:return X-s,Y+.02*s*s,YAW+(.12*s)
        return X-s,Y+math.tan(math.pi-YAW)*s,YAW
    # Helper routine: project; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def project(_route,x,y):
        return (max(0.,X-x),0.,YAW)
    # Helper routine: rectangle; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def rectangle(x,y,yaw,length,width):
        return box(x-length/2,y-width/2,x+length/2,y+width/2)
    return route,road,300,pose,project,rectangle


# Test class TestM6B5: groups related setup, helpers, and assertions behind one fixture lifecycle.
class TestM6B5(unittest.TestCase):
    # Fixture lifecycle: create or release the temporary state used only by this test class.
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.dir=Path(self.tmp.name)
        self.obs={'body_center_cr_xy_m':[X,Y], 'yaw_cr_rad':YAW,
                  'sha256':'stable-observation-hash','source_frame':967030}
        self.model={'schema_version':VERSION,'status':'provisional_offline_fit',
          'observation_sha256':self.obs['sha256'],'manual_gear':1,
          'automatic_mode_validated':False,'carla_command_to_commonroad_steering_sign':-1,
          'carla_map':'Carla/Maps/Town01','geometric_wheelbase_m':3.004648862774714,
          'front_wheel_signed_deg_per_normalized_steer':66.57350148473468,
          'calibrated_steer_command_abs_max':.08}
        self.mp=self.dir/'model.json';self.mp.write_text(json.dumps(self.model))
        self.routep=self.dir/'route.csv';self.routep.write_text('CommonRoad-export fixture')
        self.mapp=self.dir/'map.xml';self.mapp.write_text('CommonRoad-lanelet fixture')
        self.sp=self.dir/'summary.json';self.sp.write_text(json.dumps({
            'experiment':'Milestone 4E-B fixture','route_sha256':digest(self.routep),
            'map_sha256':digest(self.mapp)}))
        self.geom=geometry()
        self.reference=ref.generate(self.obs,self.model,self.mp,self.routep,self.mapp,self.sp,self.geom)
        self.refp=self.dir/'ref.json';self.refp.write_text(json.dumps(self.reference))
    # Fixture lifecycle: create or release the temporary state used only by this test class.
    def tearDown(self):self.tmp.cleanup()
    # Test method: verifies fully new route reference without changing the behavior under test.
    def test_fully_new_route_reference(self):
        a=self.reference
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(a['count'],85)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(a['reference_source'].startswith('CommonRoad-converted'))
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertFalse(a['original_rulebook_planner_used'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertNotIn('empirical_trajectory_sha256',a)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertAlmostEqual(a['offline_feasibility']['reference_displacement_station_m'],2.186,places=3)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(a['offline_feasibility']['maximum_sampled_outside_road_area_m2'],0.)
    # Test method: verifies speed and steering are bounded without changing the behavior under test.
    def test_speed_and_steering_are_bounded(self):
        rows=self.reference['samples']
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertLess(max(r['reference_speed_mps'] for r in rows),1.65)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(all(abs(r['feedforward_steer_normalized'])<=.08 for r in rows))
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(rows[-1]['reference_speed_mps'],0.)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(rows[36]['command_throttle'],.2)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(rows[45]['command_brake'],.65)
    # Test method: verifies regeneration matches without changing the behavior under test.
    def test_regeneration_matches(self):
        x=ref.load_reference(self.refp,self.obs,self.model,self.mp,self.routep,self.mapp,self.sp,self.geom)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(x,self.reference)
    # Test method: verifies rejects tamper without changing the behavior under test.
    def test_rejects_tamper(self):
        altered=copy.deepcopy(self.reference)
        altered['samples'][40]['x_cr_m']+=.2
        self.refp.write_text(json.dumps(altered))
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaisesRegex(ValueError,'differs'):ref.load_reference(self.refp,self.obs,self.model,self.mp,self.routep,self.mapp,self.sp,self.geom)
    # Test method: verifies wrong map hash rejected without changing the behavior under test.
    def test_wrong_map_hash_rejected(self):
        self.mapp.write_text('changed')
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaisesRegex(ValueError,'do not match'):ref.generate(self.obs,self.model,self.mp,self.routep,self.mapp,self.sp,self.geom)
    # Test method: verifies wrong route hash rejected without changing the behavior under test.
    def test_wrong_route_hash_rejected(self):
        self.routep.write_text('changed')
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(ValueError):ref.generate(self.obs,self.model,self.mp,self.routep,self.mapp,self.sp,self.geom)
    # Test method: verifies bad 4eb experiment rejected without changing the behavior under test.
    def test_bad_4eb_experiment_rejected(self):
        self.sp.write_text(json.dumps({'experiment':'unknown','route_sha256':digest(self.routep),'map_sha256':digest(self.mapp)}))
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(ValueError):ref.generate(self.obs,self.model,self.mp,self.routep,self.mapp,self.sp,self.geom)
    # Test method: verifies bad model rejected without changing the behavior under test.
    def test_bad_model_rejected(self):
        bad=dict(self.model,carla_command_to_commonroad_steering_sign=1)
        self.mp.write_text(json.dumps(bad))
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(ValueError):ref.validated_model(self.mp,self.obs['sha256'])
    # Test method: verifies wrong observation rejected without changing the behavior under test.
    def test_wrong_observation_rejected(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(ValueError):ref.validated_model(self.mp,'not-the-observation')
    # Test method: verifies curvature unvalidated rejected without changing the behavior under test.
    def test_curvature_unvalidated_rejected(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaisesRegex(ValueError,'curvature|heading'):
            ref.generate(self.obs,self.model,self.mp,self.routep,self.mapp,self.sp,geometry(curved=True))
    # Test method: verifies offroad rectangle rejected without changing the behavior under test.
    def test_offroad_rectangle_rejected(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaisesRegex(ValueError,'road polygon'):
            ref.generate(self.obs,self.model,self.mp,self.routep,self.mapp,self.sp,geometry(narrow=True))
    # Test method: verifies origin misalignment rejected without changing the behavior under test.
    def test_origin_misalignment_rejected(self):
        bad=dict(self.obs,body_center_cr_xy_m=[X,Y+.2])
        # Helper routine: badproject; exposes a focused operation or synthetic fixture while keeping test intent readable.
        def badproject(route,x,y):return 0.,.2,YAW
        route,road,n,pose,proj,rectangle=self.geom
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaisesRegex(ValueError,'origin'):
            ref.generate(bad,self.model,self.mp,self.routep,self.mapp,self.sp,
                        (route,road,n,pose,badproject,rectangle))
    # Test method: verifies no route length rejected without changing the behavior under test.
    def test_no_route_length_rejected(self):
        route,road,n,pose,proj,rectangle=self.geom
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(ValueError):ref.generate(self.obs,self.model,self.mp,self.routep,self.mapp,self.sp,
                                                       (route[:1]+[(1.,X-1,Y)],road,n,pose,proj,rectangle))
    # Test method: verifies command schedule at boundaries without changing the behavior under test.
    def test_command_schedule_at_boundaries(self):
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(ref.actuator_schedule(22)[0],'throttle_straight')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(ref.actuator_schedule(23)[0],'route_follow_throttle')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(ref.actuator_schedule(36)[0],'route_follow_throttle')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(ref.actuator_schedule(37)[0],'coast')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(ref.actuator_schedule(46)[0],'brake')
    # Test method: verifies nominal profile is rest to rest without changing the behavior under test.
    def test_nominal_profile_is_rest_to_rest(self):
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(ref.nominal_speed(0),0)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertAlmostEqual(ref.nominal_speed(1.8),1.62)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(ref.nominal_speed(4.2),0)
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(ValueError):ref.nominal_speed(-1)
    # Test method: verifies absolute error is not rebased without changing the behavior under test.
    def test_absolute_error_is_not_rebased(self):
        pose={'x_cr_m':X+.03,'y_cr_m':Y,'yaw_cr_rad':YAW}
        r=track.point_error(pose,self.reference['samples'][0])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertAlmostEqual(r['position_error_m'],.03)
    # Test method: verifies feedback sign and clamp without changing the behavior under test.
    def test_feedback_sign_and_clamp(self):
        rr=self.reference['samples'][40]
        err={'lateral_error_m':.02,'heading_error_deg':1.0}
        on=track.command_for_reference(rr,err,self.model,True)
        off=track.command_for_reference(rr,err,self.model,False)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertNotEqual(on[1],off[1]);self.assertEqual(off[-1],0.)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertLessEqual(abs(on[1]),.08)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertLessEqual(abs(on[-1]),.02+1e-9)
    # Test method: verifies drive confirmation double opt in without changing the behavior under test.
    def test_drive_confirmation_double_opt_in(self):
        # Repeated scenario: exercise the related states independently and retain each assertion result.
        for a in (['--execute-track','--feedback','on'],['--confirm-drive']):
            # Context/exception scope: install the temporary mock or exception guard needed only for this check.
            with self.assertRaises(SystemExit) as e:track.main(a)
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertEqual(e.exception.code,2)
    # Test method: verifies offline reference main no carla and no overwrite without changing the behavior under test.
    def test_offline_reference_main_no_carla_and_no_overwrite(self):
        argv=['--observation',str(self.dir/'obs.json'),'--model',str(self.mp),
              '--route',str(self.routep),'--map',str(self.mapp),
              '--4eb-summary',str(self.sp),'--out',str(self.dir/'generated.json')]
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with patch('carla_route_reference_m6b5.b.read_stable_observation',return_value=self.obs),\
             patch('carla_route_reference_m6b5.geometry_from_project',return_value=self.geom),\
             patch('carla_route_reference_m6b5.b.preflight',side_effect=AssertionError('offline must not call CARLA')):
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertEqual(ref.main(argv),0)
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertEqual(ref.main(argv),2)
    # Test method: verifies tracking offline validator no carla without changing the behavior under test.
    def test_tracking_offline_validator_no_carla(self):
        argv=['--observation',str(self.dir/'obs.json'),'--model',str(self.mp),
              '--route',str(self.routep),'--map',str(self.mapp),
              '--4eb-summary',str(self.sp),'--reference',str(self.refp)]
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with patch('carla_route_track_m6b5.b.read_stable_observation',return_value=self.obs),\
             patch('carla_route_track_m6b5.gen.geometry_from_project',return_value=self.geom),\
             patch('carla_route_track_m6b5.b.preflight',side_effect=AssertionError('offline must not call CARLA')):
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertEqual(track.main(argv),0)


# Test class TestM6B5MockDrive: groups related setup, helpers, and assertions behind one fixture lifecycle.
class TestM6B5MockDrive(unittest.TestCase):
    # Reuse the parent test class fixture lifecycle so derived mock-drive tests receive identical setup and cleanup.
    setUp=TestM6B5.setUp
    # Reuse the parent test class fixture lifecycle so derived mock-drive tests receive identical setup and cleanup.
    tearDown=TestM6B5.tearDown
    # Helper routine: run mock; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def run_mock(self,feedback='off',stationary=False,collision_at=None,fail_restore=False,offset=0):
        world=FakeWorld();world.inject_collision_at=collision_at;world.fail_restore=fail_restore
        # Helper routine: sample; exposes a focused operation or synthetic fixture while keeping test intent readable.
        def sample(world,actor,carla,bb,expected_frame=None):
            j=0 if stationary else min(world.drive_ticks,84)
            p=self.reference['samples'][j]
            return {'frame':world.frame,'t_sim_s':world.sim_t,
                'x_cr_m':p['x_cr_m']+offset,'y_cr_m':p['y_cr_m'],'yaw_cr_rad':p['yaw_cr_rad'],
                'forward_speed_mps':0 if stationary else p['reference_speed_mps'],
                'lateral_speed_mps':0.,'vertical_speed_mps':0.,
                'total_speed_mps':0 if stationary else p['reference_speed_mps'],
                'wheel_front_left_deg':0.,'wheel_front_right_deg':0.}
        args=NS(timeout=10.,feedback=feedback,out=self.dir/('mock_'+feedback),model=self.mp,reference=self.refp)
        wp=NS(location=NS(z=0.))
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with patch('carla_route_track_m6b5.b.preflight',return_value=(FakeCarla,None,world,FakeBlueprint(),wp,{'mock':True})),\
             patch('carla_route_track_m6b5.b._get_tick',side_effect=lambda w,timeout_s=None:w.tick()),\
             patch('carla_route_track_m6b5.b.sample_actor',side_effect=sample):
            rc=track.execute(args,self.obs,self.model,self.reference,self.geom)
        return rc,json.loads((args.out/'summary.json').read_text()),world,args.out
    # Test method: verifies clean mock drive off without changing the behavior under test.
    def test_clean_mock_drive_off(self):
        rc,r,world,out=self.run_mock()
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(rc,0);self.assertEqual(r['status'],'completed')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(r['tracking_metrics']['max_abs_feedback_correction'],0)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(r['settings_restored']);self.assertTrue(r['cleanup_ok'])
    # Test method: verifies clean mock drive on without changing the behavior under test.
    def test_clean_mock_drive_on(self):
        rc,r,w,p=self.run_mock('on')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(rc,0);self.assertEqual(r['status'],'completed')
    # Test method: verifies no motion abort without changing the behavior under test.
    def test_no_motion_abort(self):
        rc,r,w,p=self.run_mock(stationary=True)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(rc,2);self.assertIn('motion',r['abort_reason'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(r['cleanup_ok'])
    # Test method: verifies collision abort without changing the behavior under test.
    def test_collision_abort(self):
        rc,r,w,p=self.run_mock(collision_at=10)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(rc,2);self.assertEqual(r['collision_count'],1);self.assertTrue(r['cleanup_ok'])
    # Test method: verifies start mismatch abort without changing the behavior under test.
    def test_start_mismatch_abort(self):
        rc,r,w,p=self.run_mock(offset=.045)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(rc,2);self.assertIn('start pose',r['abort_reason'])
    # Test method: verifies restore failure not success without changing the behavior under test.
    def test_restore_failure_not_success(self):
        rc,r,w,p=self.run_mock(fail_restore=True)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(rc,2);self.assertEqual(r['status'],'cleanup_or_restore_failed')
    # Test method: verifies road violation fails before movement without changing the behavior under test.
    def test_road_violation_fails_before_movement(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(RuntimeError):track.evaluate_road({'x_cr_m':X,'y_cr_m':Y+.8,'yaw_cr_rad':YAW},geometry(narrow=True)[1],geometry(narrow=True)[-1])
    # Test method: verifies csv metric audit without changing the behavior under test.
    def test_csv_metric_audit(self):
        _,r,w,p=self.run_mock()
        self.reference['_loaded_from']=str(self.refp)
        audit=track.audited_rows(p/'summary.json',r,self.reference)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertAlmostEqual(audit['rms_position_error_m'],r['tracking_metrics']['rms_position_error_m'])
    # Test method: verifies csv tampering rejected without changing the behavior under test.
    def test_csv_tampering_rejected(self):
        _,r,w,p=self.run_mock()
        csvpath=p/'calibration_samples.csv'
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with csvpath.open(newline='',encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
        rows[20]['position_error_m']='0.03'
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with csvpath.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(ValueError):track.audited_rows(p/'summary.json',r,self.reference)
    # Test method: verifies paired successful runs audit without unsupported claims without changing the behavior under test.
    def test_paired_successful_runs_audit_without_unsupported_claims(self):
        _,a,_,on=self.run_mock('on')
        _,z,_,off=self.run_mock('off')
        self.reference['_loaded_from']=str(self.refp)
        result=track.compare(on/'summary.json',off/'summary.json',self.dir/'pair.json',self.reference)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(result['conclusion'],'inconclusive_small_off_error')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(result['same_independent_commonroad_route_reference'])

    # Test method: verifies reject mismatched source pair without changing the behavior under test.
    def test_reject_mismatched_source_pair(self):
        _,a,_,on=self.run_mock('on')
        _,z,_,off=self.run_mock('off')
        rpath=off/'summary.json'; r=json.loads(rpath.read_text());r['route_sha256']='different'
        rpath.write_text(json.dumps(r))
        self.reference['_loaded_from']=str(self.refp)
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(ValueError):track.compare(on/'summary.json',rpath,self.dir/'pair.json',self.reference)

    # Test method: verifies pair requires distinct reports without changing the behavior under test.
    def test_pair_requires_distinct_reports(self):
        _,r,w,p=self.run_mock()
        self.reference['_loaded_from']=str(self.refp)
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(ValueError):track.compare(p/'summary.json',p/'summary.json',self.dir/'comparison.json',self.reference)

# Conditional branch: distinguish the expected state or exceptional edge before continuing validation.
if __name__=='__main__':unittest.main()
