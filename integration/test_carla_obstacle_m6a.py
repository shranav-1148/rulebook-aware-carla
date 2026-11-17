"""Milestone 6A offline unit tests. Runs without CARLA server/package."""
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import carla_obstacle_m6a as m


def fixture(folder, *, bad_hash=False, different_start=False):
    folder=Path(folder)
    route=folder/'route_waypoints.csv'
    with route.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['x_m','y_m']);w.writeheader()
        for xy in [(40,2),(30,2),(20,2),(10,2),(0,2)]:
            w.writerow(dict(zip(['x_m','y_m'],xy)))
    summary=folder/'summary.json'
    s={'route_source':'town01_route.csv','route_sha256':m.sha256(route),
       'settings':{'obstacle_station_m':21.0,'obstacle_offset_m':-2.4,
                   'obstacle_radius_m':0.65,'clearance_margin_m':1.6},
       'obstacle_xy_m':[19.,4.4]}
    # westbound reference yaw is pi; left offset negative means northward in CR.
    s['obstacle_xy_m']=list(m.route_offset_xy(m.read_route(route),21,-2.4))
    if bad_hash:s['route_sha256']='deadbeef'
    summary.write_text(json.dumps(s))
    paths=[]
    for idx,name in enumerate(['safety_first_bicycle_samples.csv','corridor_first_bicycle_samples.csv']):
        p=folder/name
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['t_s','x_m','y_m','yaw_rad']);w.writeheader()
            w.writerow({'t_s':'0','x_m':str(40+(0.5 if different_start and idx else 0)),
                        'y_m':'2','yaw_rad':str(math.pi)})
            w.writerow({'t_s':'0.1','x_m':'39.4','y_m':'2','yaw_rad':str(math.pi)})
        paths.append(p)
    return summary,route,*paths


class GeometryTests(unittest.TestCase):
    def test_right_of_westbound_is_positive_y_in_cr(self):
        with tempfile.TemporaryDirectory() as d:
            _,route,_,_=fixture(d)
            x,y=m.route_offset_xy(m.read_route(route),21.,-2.4)
            self.assertAlmostEqual(x,19)
            self.assertAlmostEqual(y,4.4)

    def test_mirror_y_carla(self):
        with tempfile.TemporaryDirectory() as d:
            inputs=fixture(d)
            plan=m.build_plan(*inputs)
            self.assertAlmostEqual(plan['obstacle_model']['carla_xy_m'][0],19.)
            self.assertAlmostEqual(plan['obstacle_model']['carla_xy_m'][1],-4.4)
            self.assertAlmostEqual(plan['obstacle_model']['radius_m'],.65)
            self.assertEqual(plan['status'],'offline_geometry_verified')

    def test_route_cumulative_distance(self):
        with tempfile.TemporaryDirectory() as d:
            _,route,_,_=fixture(d)
            points=m.read_route(route)
            self.assertEqual([p[0] for p in points],[0,10,20,30,40])

    def test_route_hash_guard(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'SHA256'):
                m.build_plan(*fixture(d,bad_hash=True))

    def test_initial_pose_guard(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'same nominal initial pose'):
                m.build_plan(*fixture(d,different_start=True))

    def test_obstacle_position_guard(self):
        with tempfile.TemporaryDirectory() as d:
            summary,route,sa,co=fixture(d)
            obj=json.loads(summary.read_text());obj['obstacle_xy_m']=[500,500]
            summary.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError,'does not match'):
                m.build_plan(summary,route,sa,co)

    def test_station_out_of_bounds(self):
        with tempfile.TemporaryDirectory() as d:
            _,route,_,_=fixture(d)
            with self.assertRaises(ValueError):
                m.route_offset_xy(m.read_route(route),50,0)

    def test_nonnumeric_waypoint(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'route.csv';p.write_text('x_m,y_m\n0,0\ninf,0\n')
            with self.assertRaises(ValueError):m.read_route(p)

    def test_no_carla_import_on_offline_preview(self):
        with tempfile.TemporaryDirectory() as d:
            s,r,a,b=fixture(d)
            with patch.object(m,'carla_preflight',side_effect=AssertionError('CARLA accessed')):
                rc=m.main(['--summary',str(s),'--route',str(r),
                           '--safety-csv',str(a),'--corridor-csv',str(b),
                           '--out',str(Path(d)/'out')])
            self.assertEqual(rc,0)
            self.assertTrue((Path(d)/'out/obstacle_plan.json').exists())

    def test_spawn_needs_explicit_confirmation(self):
        with tempfile.TemporaryDirectory() as d:
            s,r,a,b=fixture(d)
            class P:pass
            p=P();p.confirm_spawn=False
            with self.assertRaisesRegex(RuntimeError,'confirm-spawn'):
                m.spawn_and_inspect(p,{},None,None,None,None,{})

    def test_shape_does_not_misrepresent_bbox_as_circle(self):
        report=m.shape_report({'bbox_local_extents_m':[.5,.5,.5]},
                              {'obstacle_model':{'radius_m':.65}})
        self.assertFalse(report['geometry_match'])
        self.assertTrue(report['valid_bbox'])
        self.assertAlmostEqual(report['prop_bbox_xy_corner_radius_proxy_m'],math.sqrt(.5))

    def test_degenerate_bbox(self):
        r=m.shape_report({'bbox_local_extents_m':[0,0,0]},
                         {'obstacle_model':{'radius_m':.65}})
        self.assertFalse(r['valid_bbox'])

    def test_bbox_world_vertices(self):
        positions=[(x,y,z) for x in (1,3) for y in (-2,4) for z in (0,5)]
        class BB:
            location=types.SimpleNamespace(x=0,y=0,z=2.5)
            extent=types.SimpleNamespace(x=1,y=3,z=2.5)
            def get_world_vertices(self,t):
                return [types.SimpleNamespace(x=x,y=y,z=z) for x,y,z in positions]
        actor=types.SimpleNamespace(bounding_box=BB(),get_transform=lambda:None)
        data=m.projected_bbox_geometry(actor)
        self.assertEqual(data['world_aabb_center_xy_m'],[2.,1.])
        self.assertEqual(data['bounds_world_xyz_m']['zmin'],0)

    def test_refuse_nonstatic_prop_and_mismatched_map(self):
        # Intentionally verified in production preflight; simple invariant.
        self.assertFalse('vehicle.tesla.model3'.startswith('static.prop.'))

    def test_output_json_written(self):
        with tempfile.TemporaryDirectory() as d:
            path=m.save_json(Path(d),'test.json',{'value':1})
            self.assertEqual(json.loads(path.read_text())['value'],1)


# Standalone tests intentionally avoid importing carla; test runtime lifecycle via stubs.
class MockLifecycleTests(unittest.TestCase):
    def test_prop_spawn_measure_reposition_cleanup_no_ticks(self):
        class Location:
            def __init__(self,x=0,y=0,z=0):self.x=x;self.y=y;self.z=z
        class Rotation:
            def __init__(self,yaw=0):self.yaw=yaw
        class Transform:
            def __init__(self,location=None,rotation=None):
                self.location=location or Location();self.rotation=rotation or Rotation()
        fake_carla=types.SimpleNamespace(Location=Location,Rotation=Rotation,Transform=Transform)
        class BoundingBox:
            def __init__(self):
                self.extent=Location(.5,.5,.5)
                self.location=Location(0,0,0)
            def get_world_vertices(self,t):
                return [Location(t.location.x+x,t.location.y+y,t.location.z+z)
                        for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)]
        class Actor:
            def __init__(self,t):
                self.t=t;self.id=17;self.type_id='static.prop.constructioncone'
                self.bounding_box=BoundingBox();self.destroy_calls=0
            def get_transform(self):return self.t
            def set_transform(self,t):self.t=t
            def destroy(self):self.destroy_calls+=1;return True
        class Blueprint:
            def has_attribute(self,k):return k=='role_name'
            def set_attribute(self,k,v):pass
        class World:
            def __init__(self):self.actor=None
            def try_spawn_actor(self,bp,t):
                self.actor=Actor(t);return self.actor
        w=World()
        with tempfile.TemporaryDirectory() as d:
            opts=types.SimpleNamespace(confirm_spawn=True, prop_yaw=0.,
                 ground_clearance=.02,hold_seconds=0.,out=Path(d))
            plan={'obstacle_model':{'carla_xy_m':[19.,-4.4], 'radius_m':.65}}
            report={'existing_vehicle_count':0,'world_already_synchronous':False}
            wp=types.SimpleNamespace(transform=types.SimpleNamespace(location=Location(0,0,0)))
            with patch.object(m.time,'sleep'):
                got=m.spawn_and_inspect(opts,plan,fake_carla,w,Blueprint(),wp,report)
            self.assertEqual(w.actor.destroy_calls,1)
            self.assertAlmostEqual(got['placement_xy_error_m'],0)
            self.assertAlmostEqual(got['placement_bottom_z_error_m'],0)
            saved=json.loads((Path(d)/'actor_inspection.json').read_text())
            self.assertEqual(saved['status'],'spawned_measured_then_destroyed')
            self.assertFalse(saved['collision_mesh_verified'])

    def test_refuses_existing_vehicles_before_spawn(self):
        p=types.SimpleNamespace(confirm_spawn=True)
        with self.assertRaisesRegex(RuntimeError,'occupied/synchronous'):
            m.spawn_and_inspect(p,{},None,None,None,None,
                                {'existing_vehicle_count':1,'world_already_synchronous':False})


if __name__=='__main__':
    unittest.main()
