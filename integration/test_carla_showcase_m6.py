"""Offline tests; no CARLA server and no physics simulation required."""
# Import dependency: __future__ supplies types or behavior used by this test module.
from __future__ import annotations
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
# Import dependency: unittest.mock supplies types or behavior used by this test module.
from unittest.mock import patch

# Import dependency: showcase_data_m6 supplies types or behavior used by this test module.
import showcase_data_m6 as data
# Import dependency: carla_showcase_m6 supplies types or behavior used by this test module.
import carla_showcase_m6 as show


# Test class ShowcaseTests: groups related setup, helpers, and assertions behind one fixture lifecycle.
class ShowcaseTests(unittest.TestCase):
    # Test method: verifies scenes exist and ordered without changing the behavior under test.
    def test_scenes_exist_and_ordered(self):
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(len(data.SCENE_ORDER), 12)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(data.SCENE_ORDER[0], 'intro')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(data.SCENE_ORDER[-1], 'outro')
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertIn('planning',data.SCENE_ORDER)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertIn('speed_fixed',data.SCENE_ORDER)

    # Test method: verifies obs rejects unsettled without changing the behavior under test.
    def test_obs_rejects_unsettled(self):
        doc={'stability_capture_mode':'passive_multi_frame','stability':{'latest':{'accepted':False}},'cleanup_ok':True,
             'ego':{'body_center_cr_xy_m':[1,2],'yaw_cr_rad':0}}
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(ValueError):data.observation_pose(doc)
        doc['stability']['latest']['accepted']=True
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(data.observation_pose(doc),data.Pose(1.,2.,0.))

    # Test method: verifies observation requires cleanup without changing the behavior under test.
    def test_observation_requires_cleanup(self):
        doc={'stability_capture_mode':'passive_multi_frame','stability':{'latest':{'accepted':True}},'cleanup_ok':False,
             'ego':{'body_center_cr_xy_m':[1,2],'yaw_cr_rad':0}}
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(ValueError):data.observation_pose(doc)

    # Test method: verifies read samples monotonic without changing the behavior under test.
    def test_read_samples_monotonic(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'t.csv'
            # Context/exception scope: install the temporary mock or exception guard needed only for this check.
            with p.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['t_s','x_cr_m','y_cr_m','yaw_cr_rad']);w.writerow([0,1,2,0]);w.writerow([0,1.1,2,0])
            # Context/exception scope: install the temporary mock or exception guard needed only for this check.
            with self.assertRaisesRegex(ValueError,'time is not increasing'):data.read_samples(p)

    # Test method: verifies read samples dt without changing the behavior under test.
    def test_read_samples_dt(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'t.csv'
            # Context/exception scope: install the temporary mock or exception guard needed only for this check.
            with p.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['t_s','x_cr_m','y_cr_m','yaw_cr_rad']);w.writerow([0,1,2,0]);w.writerow([0.1,1.1,2,0])
            # Context/exception scope: install the temporary mock or exception guard needed only for this check.
            with self.assertRaisesRegex(ValueError,'expected 0.05s timestep'):data.read_samples(p)

    # Test method: verifies poses switch column names without changing the behavior under test.
    def test_poses_switch_column_names(self):
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(data.pose({'x_cr_m':'1','y_cr_m':'2','yaw_cr_rad':'-3'}),data.Pose(1,2,-3))
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(data.pose({'x_m':'1','y_m':'2','yaw_rad':'-3'},planning=True),data.Pose(1,2,-3))

    # Test method: verifies plan status must be offline without changing the behavior under test.
    def test_plan_status_must_be_offline(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(ValueError):data.check_plan_report({'status':'offline_plan_generated','executed_carla_control':True,
           'divergence':{'max_time_aligned_separation_m':5}})
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(ValueError):data.check_plan_report({'status':'offline_plan_generated','executed_carla_control':False,
           'divergence':{'max_time_aligned_separation_m':0}})

    # Helper routine: make source root; exposes a focused operation or synthetic fixture while keeping test intent readable.
    def _make_source_root(self, root):
        obs=root/data.ROOT_RELATIVE_PATHS['observation'];obs.parent.mkdir(parents=True)
        obs.write_text(json.dumps({'stability_capture_mode':'passive_multi_frame','cleanup_ok':True,
            'stability':{'latest':{'accepted':True}},
            'ego':{'body_center_cr_xy_m':[385,2],'yaw_cr_rad':math.pi},
            'obstacle':{'center_cr_xy_m':[363,4.4]}}))
        path=root/data.ROOT_RELATIVE_PATHS['straight'];path.parent.mkdir(parents=True)
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['t_s','x_cr_m','y_cr_m','yaw_cr_rad'])
            writer.writeheader()
            # Repeated scenario: exercise the related states independently and retain each assertion result.
            for i in range(85): writer.writerow({'t_s':round(i*.05,5),'x_cr_m':385-i*.02,'y_cr_m':2,'yaw_cr_rad':math.pi})

    # Test method: verifies validate offline straight without changing the behavior under test.
    def test_validate_offline_straight(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with tempfile.TemporaryDirectory() as td:
            self._make_source_root(Path(td))
            result=data.validate_sources(td,['intro','straight','outro'])
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertEqual(result['scenes'],['intro','straight','outro'])
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertIn('observation',result['source_hashes'])
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertEqual(len(result['data']['straight']),85)

    # Test method: verifies reject missing archive by default without changing the behavior under test.
    def test_reject_missing_archive_by_default(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with tempfile.TemporaryDirectory() as td:
            self._make_source_root(Path(td))
            # Context/exception scope: install the temporary mock or exception guard needed only for this check.
            with self.assertRaises(FileNotFoundError):data.validate_sources(td,['straight','steer_neg'])
            partial=data.validate_sources(td,['straight','steer_neg'],skip_missing=True)
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertEqual(partial['scenes'],['straight'])

    # Test method: verifies ensure stable obs for intro without changing the behavior under test.
    def test_ensure_stable_obs_for_intro(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with tempfile.TemporaryDirectory() as td:
            # Context/exception scope: install the temporary mock or exception guard needed only for this check.
            with self.assertRaises(FileNotFoundError):data.validate_sources(td,['intro'])

    # Test method: verifies archive time index without changing the behavior under test.
    def test_archive_time_index(self):
        arr=[{'t_s':i*.05} for i in range(43)]
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(show.lookup_scene_index('speed_failed',{'speed_failed':arr},2.1),42)
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaises(ValueError):show.lookup_scene_index('speed_failed',{'speed_failed':arr},3.)

    # Test method: verifies caption sources present without changing the behavior under test.
    def test_caption_sources_present(self):
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with tempfile.TemporaryDirectory() as td:
            self._make_source_root(Path(td))
            b=data.validate_sources(td,['straight'])
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertEqual(data.scene_summary('straight',b['data'])[2],
              'RECORDED CARLA TRAJECTORY RECONSTRUCTION — not a new drive')

    # Test method: verifies no unphysical synthetic scene data without changing the behavior under test.
    def test_no_unphysical_synthetic_scene_data(self):
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertTrue(all(isinstance(s,list) for s in data.SCENE_KEYS.values()))
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertIn('plan_report',data.SCENE_KEYS['planning'])
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertNotIn('plan_report',data.SCENE_KEYS['route_on'])

    # Test method: verifies kinematic placement mirror sign and origin without changing the behavior under test.
    def test_kinematic_placement_mirror_sign_and_origin(self):
        # Test class Loc: groups related setup, helpers, and assertions behind one fixture lifecycle.
        class Loc:
            # Helper routine: init; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def __init__(self,*a,**k):self.x=k.get('x',a[0] if len(a)>0 else 0);self.y=k.get('y',a[1] if len(a)>1 else 0);self.z=k.get('z',a[2] if len(a)>2 else 0)
        # Test class Rot: groups related setup, helpers, and assertions behind one fixture lifecycle.
        class Rot:
            # Helper routine: init; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def __init__(self,*a,**k):self.yaw=k.get('yaw',0);self.pitch=k.get('pitch',0)
        # Test class Tf: groups related setup, helpers, and assertions behind one fixture lifecycle.
        class Tf:
            # Helper routine: init; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def __init__(self,l,r):self.location=l;self.rotation=r
        # Test class Carla: groups related setup, helpers, and assertions behind one fixture lifecycle.
        class Carla:Location=Loc;Rotation=Rot;Transform=Tf
        p=data.Pose(385,2,math.pi)
        tf=show.location_and_yaw(Carla,p,.1)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertAlmostEqual(tf.location.y,-2)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertAlmostEqual(tf.rotation.yaw,-180)
        bb=type('B',(),{'location':Loc(1,0,0)})()
        shifted=show.actor_origin(Carla,p,.1,bb)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertAlmostEqual(shifted.location.x,386)

    # Test method: verifies reject occupied and sync world without changing the behavior under test.
    def test_reject_occupied_and_sync_world(self):
        # Test class Settings: groups related setup, helpers, and assertions behind one fixture lifecycle.
        class Settings:synchronous_mode=False
        # Test class FakeMap: groups related setup, helpers, and assertions behind one fixture lifecycle.
        class FakeMap:name='Carla/Maps/Town01'
        # Test class W: groups related setup, helpers, and assertions behind one fixture lifecycle.
        class W:
            # Helper routine: init; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def __init__(self,actors):self.actors=actors;self.settings=Settings()
            # Helper routine: get map; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def get_map(self):return FakeMap()
            # Helper routine: get settings; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def get_settings(self):return self.settings
            # Helper routine: get actors; exposes a focused operation or synthetic fixture while keeping test intent readable.
            def get_actors(self):return self.actors
        # Test class A: groups related setup, helpers, and assertions behind one fixture lifecycle.
        class A:type_id='vehicle.tesla.model3'
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaisesRegex(RuntimeError,'not isolated'):
            show.collision_free_preflight(None,W([A()]),None)
        w=W([]);w.settings.synchronous_mode=True
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with self.assertRaisesRegex(RuntimeError,'already synchronous'):
            show.collision_free_preflight(None,w,None)

    # Test method: verifies movie encoder small frame without changing the behavior under test.
    def test_movie_encoder_small_frame(self):
        # Import dependency: PIL supplies types or behavior used by this test module.
        from PIL import Image
        # Context/exception scope: install the temporary mock or exception guard needed only for this check.
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'t.mp4'
            clip=show.Film(p,96,64,2)
            clip.send(Image.new('RGB',(96,64),(50,80,100)).tobytes())
            clip.close()
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertTrue(p.is_file())
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertGreater(p.stat().st_size,300)
            # Observable assertion: compare the resulting value or state with the contract documented by this test.
            self.assertEqual(clip.frames,2)

    # Test method: verifies actual uploaded csv schema without changing the behavior under test.
    def test_actual_uploaded_csv_schema(self):
        # Optional local check when the attached saved CSV exists; never required on user machine.
        sample=Path('/mnt/data/calibration_samples(1).csv')
        # Conditional branch: distinguish the expected state or exceptional edge before continuing validation.
        if not sample.exists():self.skipTest('No uploaded sample fixture')
        rows=data.read_samples(sample)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertEqual(len(rows),85)
        # Observable assertion: compare the resulting value or state with the contract documented by this test.
        self.assertGreater(data.pose(rows[-1]).x,0)


# Conditional branch: distinguish the expected state or exceptional edge before continuing validation.
if __name__=='__main__':unittest.main()
