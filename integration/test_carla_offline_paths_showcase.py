"""Offline tests for the offline-planned-path cinematic recorder; no CARLA server."""
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import carla_offline_paths_showcase as demo
import showcase_data_m6 as src


class PlannerShowcaseTests(unittest.TestCase):
    def synthetic_root(self,root):
        files={k:root/v for k,v in src.ROOT_RELATIVE_PATHS.items()
               if k in ('observation','plan_report','plan_safety','plan_corridor')}
        for f in files.values():f.parent.mkdir(parents=True,exist_ok=True)
        obs={'stability_capture_mode':'passive_multi_frame','cleanup_ok':True,
             'stability':{'latest':{'accepted':True}},
             'ego':{'body_center_cr_xy_m':[385.,2.], 'yaw_cr_rad':math.pi},
             'obstacle':{'center_cr_xy_m':[363.,4.4]}}
        files['observation'].write_text(json.dumps(obs))
        fields=['t_s','x_m','y_m','yaw_rad','speed_mps','station_m','offset_m']
        for mode in ('safety','corridor'):
            with files['plan_'+mode].open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                for i in range(81):
                    t=i*.1
                    writer.writerow({'t_s':t,'x_m':385.-t*3,'y_m':2.+(0.8*t/8 if mode=='safety' else 0),
                       'yaw_rad':math.pi,'speed_mps':3,'station_m':3*t,'offset_m':0})
        rep={'status':'offline_plan_generated','executed_carla_control':False,
             'divergence':{'max_time_aligned_separation_m':.8},
             'settings':{'horizon_steps':8,'dt_s':1},
             'profiles':{'safety_first':{},'corridor_first':{}}}
        files['plan_report'].write_text(json.dumps(rep))
        return files

    def test_bundle_only_requires_planner_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.synthetic_root(root)
            b=demo.check_bundle(root)
            self.assertEqual(b['scenes'],['planning'])
            self.assertEqual(set(b['data']),{'observation','plan_report','plan_safety','plan_corridor'})

    def test_reject_missing_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);files=self.synthetic_root(root)
            files['plan_corridor'].unlink()
            with self.assertRaises(FileNotFoundError):demo.check_bundle(root)

    def test_reject_mismatching_divergence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);files=self.synthetic_root(root)
            report=json.loads(files['plan_report'].read_text())
            report['divergence']['max_time_aligned_separation_m']=3.
            files['plan_report'].write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'do not reproduce reported'):
                demo.check_bundle(root)

    def test_interpolates_at_20hz_from_10hz_samples(self):
        rows=[{'t_s':'0','x_m':'0','y_m':'0','yaw_rad':'0'},
              {'t_s':'0.1','x_m':'1','y_m':'1','yaw_rad':'0.2'}]
        p=demo.sample_plan(rows,.05)
        self.assertAlmostEqual(p.x,.5)
        self.assertAlmostEqual(p.y,.5)
        self.assertAlmostEqual(p.yaw,.1)

    def test_yaw_interpolation_near_pi_boundary(self):
        rows=[{'t_s':'0','x_m':'0','y_m':'0','yaw_rad':str(math.radians(179))},
              {'t_s':'0.1','x_m':'1','y_m':'1','yaw_rad':str(math.radians(-179))}]
        p=demo.sample_plan(rows,.05)
        self.assertAlmostEqual(abs(p.yaw),math.pi,places=6)

    def test_original_ends_exactly_before_extension(self):
        rows=[{'t_s':'0','x_m':'10','y_m':'2','yaw_rad':'0'},
              {'t_s':'0.1','x_m':'11','y_m':'2','yaw_rad':'0'}]
        sequence=demo.make_playback(rows,2.,1.)
        original,extended=demo.ghost_lines(sequence)
        self.assertEqual(len(original),3)
        self.assertEqual(len(extended),41)
        self.assertTrue(original[-1].original)
        self.assertAlmostEqual(original[-1].x,11.)
        self.assertFalse(extended[1].original)
        self.assertAlmostEqual(extended[-1].x,13.)
        self.assertAlmostEqual(extended[-1].extension_m,2.)

    def test_default_extension_more_distance_than_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.synthetic_root(root);bundle=demo.check_bundle(root)
            args=type('A',(),{'extra_distance_m':14.,'extension_speed_mps':1.4})()
            paths=demo.make_sequence(bundle,args)
            self.assertGreater(len(paths['safety']),len(bundle['data']['plan_safety']))
            self.assertAlmostEqual(paths['safety'][-1].extension_m,14.)
            self.assertAlmostEqual(paths['corridor'][-1].extension_m,14.)
            self.assertEqual(sum(not p.original for p in paths['safety']),200)

    def test_zero_extension_is_only_true_source(self):
        rows=[{'t_s':'0','x_m':'2','y_m':'2','yaw_rad':'0'},
              {'t_s':'0.1','x_m':'3','y_m':'2','yaw_rad':'0'}]
        path=demo.make_playback(rows,0.,1.)
        self.assertTrue(all(p.original for p in path))
        original,extra=demo.ghost_lines(path)
        self.assertEqual(len(extra),1)
        self.assertEqual(original[-1],extra[0])

    def test_bad_extension_rejected(self):
        rows=[{'t_s':'0','x_m':'2','y_m':'2','yaw_rad':'0'},
              {'t_s':'0.1','x_m':'3','y_m':'2','yaw_rad':'0'}]
        for dist,speed in [(-1,1),(81,1),(1,0),(1,5.1)]:
            with self.subTest(dist=dist,speed=speed):
                with self.assertRaises(ValueError):demo.make_playback(rows,dist,speed)

    def test_colours_not_bright_red_green(self):
        s=demo.COLORS['safety'];c=demo.COLORS['corridor']
        self.assertTrue(all(90 <= x <= 208 for x in s+c))
        self.assertNotEqual(s,c)

    def test_preflight_refuses_occupied_or_synchronous(self):
        class Settings:synchronous_mode=False
        class Map:name='Carla/Maps/Town01'
        class World:
            def __init__(self,actors):self.actors=actors;self.settings=Settings()
            def get_map(self):return Map()
            def get_settings(self):return self.settings
            def get_actors(self):return self.actors
        class Actor:type_id='sensor.camera.rgb'
        with self.assertRaisesRegex(RuntimeError,'not isolated'):
            demo.preflight(World([Actor()]))
        w=World([])
        w.settings.synchronous_mode=True
        with self.assertRaisesRegex(RuntimeError,'already synchronous'):
            demo.preflight(w)
        w.settings.synchronous_mode=False
        self.assertEqual(demo.preflight(w)['other_vehicles_walkers_sensors'],0)

    def test_pose_conversion_body_center_and_mirror(self):
        class Loc:
            def __init__(self,**kw):self.x=kw.get('x',0);self.y=kw.get('y',0);self.z=kw.get('z',0)
        class Rot:
            def __init__(self,**kw):self.yaw=kw.get('yaw',0)
        class Tf:
            def __init__(self,loc,rot):self.location=loc;self.rotation=rot
        class Carla:Location=Loc;Rotation=Rot;Transform=Tf
        p=demo.PlaybackPoint(385.,2.,math.pi,1.,True)
        bb=type('B',(),{'location':Loc(x=1.35)})()
        t=demo.to_carla_transform(Carla,p,0.4,bb)
        self.assertAlmostEqual(t.location.x,386.35)
        self.assertAlmostEqual(t.location.y,-2)
        self.assertAlmostEqual(t.rotation.yaw,-180)

    def test_caption_renders_over_fake_camera_image(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.synthetic_root(root)
            bundle=demo.check_bundle(root)
            args=type('A',(),{'extra_distance_m':1.,'extension_speed_mps':1.})()
            paths=demo.make_sequence(bundle,args)
            frame=paths['safety'][0]
            img=Image.new('RGBA',(960,540),(30,55,70,255))
            # CARLA image.raw_data is BGRA bytes
            raw=img.tobytes('raw','BGRA')
            image=type('I',(),{'width':960,'height':540,'raw_data':raw})()
            screen=demo.add_captions(image,'safety',2,3,frame,paths,(363,4.4),
                 ['foo/safety.csv','foo/corridor.csv'],bundle['data']['plan_report'],2)
            self.assertEqual(len(screen),960*540*3)
            screen2=demo.add_captions(image,'overview',1,3,None,paths,(363,4.4),
                 ['foo/safety.csv','foo/corridor.csv'],bundle['data']['plan_report'],2)
            self.assertEqual(len(screen2),len(screen))
            ext_frame=paths['safety'][-1]
            screen3=demo.add_captions(image,'safety',2,3,ext_frame,paths,(363,4.4),
                 ['foo/safety.csv','foo/corridor.csv'],bundle['data']['plan_report'],2)
            self.assertNotEqual(screen,screen3)

    def test_raw_camera_no_overlay_preserves_rgb_pixels(self):
        from PIL import Image
        img=Image.new('RGB',(8,4),(23,78,151))
        raw=img.convert('RGBA').tobytes('raw','BGRA')
        image=type('I',(),{'width':8,'height':4,'raw_data':raw})()
        self.assertEqual(demo.camera_rgb_bytes(image),img.tobytes())

    def test_no_overlay_flag_in_parser(self):
        import unittest.mock as mock
        with mock.patch.object(demo,'check_bundle',side_effect=RuntimeError('source check sentinel')):
            # Parser must accept --no-overlay (no CARLA invocation).
            rc=demo.main(['--check','--no-overlay'])
        self.assertEqual(rc,2)

    def test_small_mp4_encoder(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'sample.mp4'
            encoder=demo.Film(path,96,64,2)
            encoder.send(Image.new('RGB',(96,64),(110,130,145)).tobytes())
            encoder.close()
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size,300)
            self.assertEqual(encoder.frames,2)

if __name__=='__main__':unittest.main()
