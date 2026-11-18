"""Offline tests; no CARLA server and no physics simulation required."""
from __future__ import annotations
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import showcase_data_m6 as data
import carla_showcase_m6 as show


class ShowcaseTests(unittest.TestCase):
    def test_scenes_exist_and_ordered(self):
        self.assertEqual(len(data.SCENE_ORDER), 12)
        self.assertEqual(data.SCENE_ORDER[0], 'intro')
        self.assertEqual(data.SCENE_ORDER[-1], 'outro')
        self.assertIn('planning',data.SCENE_ORDER)
        self.assertIn('speed_fixed',data.SCENE_ORDER)

    def test_obs_rejects_unsettled(self):
        doc={'stability_capture_mode':'passive_multi_frame','stability':{'latest':{'accepted':False}},'cleanup_ok':True,
             'ego':{'body_center_cr_xy_m':[1,2],'yaw_cr_rad':0}}
        with self.assertRaises(ValueError):data.observation_pose(doc)
        doc['stability']['latest']['accepted']=True
        self.assertEqual(data.observation_pose(doc),data.Pose(1.,2.,0.))

    def test_observation_requires_cleanup(self):
        doc={'stability_capture_mode':'passive_multi_frame','stability':{'latest':{'accepted':True}},'cleanup_ok':False,
             'ego':{'body_center_cr_xy_m':[1,2],'yaw_cr_rad':0}}
        with self.assertRaises(ValueError):data.observation_pose(doc)

    def test_read_samples_monotonic(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'t.csv'
            with p.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['t_s','x_cr_m','y_cr_m','yaw_cr_rad']);w.writerow([0,1,2,0]);w.writerow([0,1.1,2,0])
            with self.assertRaisesRegex(ValueError,'time is not increasing'):data.read_samples(p)

    def test_read_samples_dt(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'t.csv'
            with p.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['t_s','x_cr_m','y_cr_m','yaw_cr_rad']);w.writerow([0,1,2,0]);w.writerow([0.1,1.1,2,0])
            with self.assertRaisesRegex(ValueError,'expected 0.05s timestep'):data.read_samples(p)

    def test_poses_switch_column_names(self):
        self.assertEqual(data.pose({'x_cr_m':'1','y_cr_m':'2','yaw_cr_rad':'-3'}),data.Pose(1,2,-3))
        self.assertEqual(data.pose({'x_m':'1','y_m':'2','yaw_rad':'-3'},planning=True),data.Pose(1,2,-3))

    def test_plan_status_must_be_offline(self):
        with self.assertRaises(ValueError):data.check_plan_report({'status':'offline_plan_generated','executed_carla_control':True,
           'divergence':{'max_time_aligned_separation_m':5}})
        with self.assertRaises(ValueError):data.check_plan_report({'status':'offline_plan_generated','executed_carla_control':False,
           'divergence':{'max_time_aligned_separation_m':0}})

    def _make_source_root(self, root):
        obs=root/data.ROOT_RELATIVE_PATHS['observation'];obs.parent.mkdir(parents=True)
        obs.write_text(json.dumps({'stability_capture_mode':'passive_multi_frame','cleanup_ok':True,
            'stability':{'latest':{'accepted':True}},
            'ego':{'body_center_cr_xy_m':[385,2],'yaw_cr_rad':math.pi},
            'obstacle':{'center_cr_xy_m':[363,4.4]}}))
        path=root/data.ROOT_RELATIVE_PATHS['straight'];path.parent.mkdir(parents=True)
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['t_s','x_cr_m','y_cr_m','yaw_cr_rad'])
            writer.writeheader()
            for i in range(85): writer.writerow({'t_s':round(i*.05,5),'x_cr_m':385-i*.02,'y_cr_m':2,'yaw_cr_rad':math.pi})

    def test_validate_offline_straight(self):
        with tempfile.TemporaryDirectory() as td:
            self._make_source_root(Path(td))
            result=data.validate_sources(td,['intro','straight','outro'])
            self.assertEqual(result['scenes'],['intro','straight','outro'])
            self.assertIn('observation',result['source_hashes'])
            self.assertEqual(len(result['data']['straight']),85)

    def test_reject_missing_archive_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            self._make_source_root(Path(td))
            with self.assertRaises(FileNotFoundError):data.validate_sources(td,['straight','steer_neg'])
            partial=data.validate_sources(td,['straight','steer_neg'],skip_missing=True)
            self.assertEqual(partial['scenes'],['straight'])

    def test_ensure_stable_obs_for_intro(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):data.validate_sources(td,['intro'])

    def test_archive_time_index(self):
        arr=[{'t_s':i*.05} for i in range(43)]
        self.assertEqual(show.lookup_scene_index('speed_failed',{'speed_failed':arr},2.1),42)
        with self.assertRaises(ValueError):show.lookup_scene_index('speed_failed',{'speed_failed':arr},3.)

    def test_caption_sources_present(self):
        with tempfile.TemporaryDirectory() as td:
            self._make_source_root(Path(td))
            b=data.validate_sources(td,['straight'])
            self.assertEqual(data.scene_summary('straight',b['data'])[2],
              'RECORDED CARLA TRAJECTORY RECONSTRUCTION — not a new drive')

    def test_no_unphysical_synthetic_scene_data(self):
        self.assertTrue(all(isinstance(s,list) for s in data.SCENE_KEYS.values()))
        self.assertIn('plan_report',data.SCENE_KEYS['planning'])
        self.assertNotIn('plan_report',data.SCENE_KEYS['route_on'])

    def test_kinematic_placement_mirror_sign_and_origin(self):
        class Loc:
            def __init__(self,*a,**k):self.x=k.get('x',a[0] if len(a)>0 else 0);self.y=k.get('y',a[1] if len(a)>1 else 0);self.z=k.get('z',a[2] if len(a)>2 else 0)
        class Rot:
            def __init__(self,*a,**k):self.yaw=k.get('yaw',0);self.pitch=k.get('pitch',0)
        class Tf:
            def __init__(self,l,r):self.location=l;self.rotation=r
        class Carla:Location=Loc;Rotation=Rot;Transform=Tf
        p=data.Pose(385,2,math.pi)
        tf=show.location_and_yaw(Carla,p,.1)
        self.assertAlmostEqual(tf.location.y,-2)
        self.assertAlmostEqual(tf.rotation.yaw,-180)
        bb=type('B',(),{'location':Loc(1,0,0)})()
        shifted=show.actor_origin(Carla,p,.1,bb)
        self.assertAlmostEqual(shifted.location.x,386)

    def test_reject_occupied_and_sync_world(self):
        class Settings:synchronous_mode=False
        class FakeMap:name='Carla/Maps/Town01'
        class W:
            def __init__(self,actors):self.actors=actors;self.settings=Settings()
            def get_map(self):return FakeMap()
            def get_settings(self):return self.settings
            def get_actors(self):return self.actors
        class A:type_id='vehicle.tesla.model3'
        with self.assertRaisesRegex(RuntimeError,'not isolated'):
            show.collision_free_preflight(None,W([A()]),None)
        w=W([]);w.settings.synchronous_mode=True
        with self.assertRaisesRegex(RuntimeError,'already synchronous'):
            show.collision_free_preflight(None,w,None)

    def test_movie_encoder_small_frame(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'t.mp4'
            clip=show.Film(p,96,64,2)
            clip.send(Image.new('RGB',(96,64),(50,80,100)).tobytes())
            clip.close()
            self.assertTrue(p.is_file())
            self.assertGreater(p.stat().st_size,300)
            self.assertEqual(clip.frames,2)

    def test_actual_uploaded_csv_schema(self):
        # Optional local check when the attached saved CSV exists; never required on user machine.
        sample=Path('/mnt/data/calibration_samples(1).csv')
        if not sample.exists():self.skipTest('No uploaded sample fixture')
        rows=data.read_samples(sample)
        self.assertEqual(len(rows),85)
        self.assertGreater(data.pose(rows[-1]).x,0)


if __name__=='__main__':unittest.main()
