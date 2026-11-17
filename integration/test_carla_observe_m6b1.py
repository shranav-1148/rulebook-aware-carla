"""Offline and mocked-CARLA tests for Milestone 6B-1 (no CARLA server needed)."""
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import carla_observe_m6b1 as m


def obj(**kw): return types.SimpleNamespace(**kw)


class BBox:
    def __init__(self, lx=0.03, ly=0, lz=0.4, ex=2.2, ey=0.95, ez=0.8):
        self.location=obj(x=lx,y=ly,z=lz)
        self.extent=obj(x=ex,y=ey,z=ez)
    def get_world_vertices(self, pose):
        angle=math.radians(pose.rotation.yaw)
        c,s=math.cos(angle),math.sin(angle)
        verts=[]
        for i in (-1,1):
            for j in (-1,1):
                for k in (-1,1):
                    a=self.location.x+i*self.extent.x
                    b=self.location.y+j*self.extent.y
                    z=self.location.z+k*self.extent.z
                    verts.append(obj(x=pose.location.x+a*c-b*s,
                                     y=pose.location.y+a*s+b*c,z=pose.location.z+z))
        return verts


class Actor:
    def __init__(self, ident, typ, pose, bbox=None, vel=None):
        self.id,self.type_id=ident,typ
        self.pose=pose
        self.bounding_box=bbox or BBox()
        self.vel=vel or obj(x=0.,y=0.,z=0.)
        self.destroyed=False
        self.steer=(0.,0.)
        self.transform_calls=0
    def get_transform(self): return self.pose
    def set_transform(self, t):
        self.pose=t
        self.transform_calls+=1
    def get_wheel_steer_angle(self, side):
        return self.steer[0] if side=='FL' else self.steer[1]
    def destroy(self):
        self.destroyed=True
        return True


def pose(x,y,z=0,yaw=-180):
    return obj(location=obj(x=x,y=y,z=z),rotation=obj(yaw=yaw,pitch=0.,roll=0.))


class FakeSnapActor:
    def __init__(self, actor): self.actor=actor
    def get_transform(self): return self.actor.pose
    def get_velocity(self): return self.actor.vel
    def get_acceleration(self): return obj(x=0.,y=0.,z=0.)


class FakeSnapshot:
    def __init__(self, actors, frame=12):
        self.actors={a.id:a for a in actors}
        self.frame=frame
        self.timestamp=obj(elapsed_seconds=3.0,delta_seconds=.05)
    def find(self, ident):
        a=self.actors.get(ident)
        return FakeSnapActor(a) if a is not None else None


class FakeWorld:
    def __init__(self):
        self.actors=[]
        self.wait_calls=0
        self.tick_calls=0
        self.can_spawn_prop=True
    def try_spawn_actor(self, bp, transform):
        if bp.id.startswith('static.prop.') and not self.can_spawn_prop:
            return None
        if bp.id.startswith('vehicle.'):
            bb=BBox(lx=.029,ly=0.,lz=.6,ex=2.2,ey=.95,ez=.8)
        else:
            bb=BBox(lx=0.,ly=0.,lz=.293,ex=.172,ey=.172,ez=.293)
        actor=Actor(200+len(self.actors),bp.id,transform,bb)
        self.actors.append(actor)
        return actor
    def get_map(self):
        return obj(get_waypoint=lambda *a,**k: obj(transform=pose(363.59,-1.99,0)))
    def wait_for_tick(self, timeout):
        self.wait_calls+=1
        return FakeSnapshot(self.actors)
    def tick(self):
        self.tick_calls+=1
        raise AssertionError('Never tick the world in Milestone 6B-1')


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context={
            'route':[(0.,384.591,1.993),(50.,334.591,1.993)],
            'settings':{'wheelbase_m':2.7,'speed_min_mps':5.,'speed_max_mps':8.,
                        'length_m':4.5,'width_m':1.9,'max_abs_route_offset_m':2.5,
                        'max_abs_heading_error_deg':35.,'max_abs_steering_deg':25.},
            'plan':{'ego_initial_commonroad_body_center_xy_m':[384.591,1.993],
                    'ego_initial_carla_body_center_xy_m':[384.591,-1.993],
                    'ego_initial_carla_yaw_deg':-180.,
                    'obstacle_model':{'commonroad_xy_m':[363.591,4.393],
                                      'carla_xy_m':[363.591,-4.393],
                                      'radius_m':.65,'soft_clearance_margin_m':1.6}}
        }
        cls.carla=obj(VehicleWheelLocation=obj(FL_Wheel='FL',FR_Wheel='FR'),
                      Location=lambda x,y,z:obj(x=x,y=y,z=z),
                      Rotation=lambda yaw:obj(yaw=yaw,pitch=0,roll=0),
                      Transform=lambda l,r:obj(location=l,rotation=r),
                      LaneType=obj(Driving='driving'))
    def make_pair(self, vx=0., wheel=(0.,0.)):
        ego=Actor(10,'vehicle.tesla.model3',pose(384.620,-1.993),vel=obj(x=vx,y=0.,z=0.))
        ego.steer=wheel
        obs=Actor(11,'static.prop.constructioncone',pose(363.591,-4.393),
                  bbox=BBox(0.,0.,.293,.172,.172,.293))
        return ego,obs,FakeSnapshot([ego,obs])

    def test_coordinate_mirror(self):
        self.assertEqual(m.cr_from_carla(3,-4),(3.,4.))
    def test_yaw_mirror(self):
        self.assertAlmostEqual(abs(m.heading_cr_from_carla(-180)),math.pi)
    def test_body_center_uses_local_box_center(self):
        t=pose(10.,1.,yaw=90.)
        x,y=m.body_center_carla(t,obj(x=.2,y=0.))
        self.assertAlmostEqual(x,10)
        self.assertAlmostEqual(y,1.2)
    def test_inverse_body_center_correction(self):
        for yaw in [-180,0,45,90]:
            lx=obj(x=.2,y=-.3)
            x,y=m.actor_origin_for_body_center(20,3,yaw,lx)
            a,b=m.body_center_carla(pose(x,y,yaw=yaw),lx)
            self.assertAlmostEqual(a,20)
            self.assertAlmostEqual(b,3)
    def test_route_projection_westbound_right_negative(self):
        p=m.project_polyline(self.context['route'],363.591,4.393)
        self.assertAlmostEqual(p['station_m'],21,places=4)
        self.assertAlmostEqual(p['signed_lateral_offset_m'],-2.4,places=4)
    def test_route_projection_outside_endpoint(self):
        p=m.project_polyline(self.context['route'],390,1.993)
        self.assertEqual(p['station_m'],0)
    def test_bbox_world_using_pose(self):
        bbox=m.bbox_world(Actor(1,'static.prop.x',pose(3,4),BBox(0,0,0,1,2,3)),pose(3,4))
        self.assertAlmostEqual(bbox['bounds_carla_xyz_m']['xmin'],2)
        self.assertAlmostEqual(bbox['dimensions_world_aabb_m'][1],4)
    def test_missing_snapshot_actor_fails(self):
        ego,_,_=self.make_pair()
        with self.assertRaises(RuntimeError):m.snapshot_actor(FakeSnapshot([]),ego)
    def test_nominal_observation_pose_and_speed_flag(self):
        ego,obs,snap=self.make_pair()
        report=m.build_observation(snap,ego,obs,self.carla,self.context)
        self.assertEqual(report['snapshot']['frame'],12)
        self.assertTrue(report['checks']['ego_pose_matches_nominal_start'])
        self.assertTrue(report['checks']['obstacle_center_matches_offline_location'])
        self.assertFalse(report['checks']['forward_speed_within_planner_bounds'])
        self.assertFalse(report['planner_initial_state_candidate']['ready_for_live_replanning'])
        self.assertAlmostEqual(report['planner_initial_state_candidate']['rear_axle_x_cr_m'],385.941,places=2)
        self.assertIn('forward_speed_within_planner_bounds',
                      report['planner_initial_state_candidate']['not_ready_reasons'])
    def test_moving_speed_candidate_still_not_ready(self):
        ego,obs,snap=self.make_pair(vx=-6.)
        r=m.build_observation(snap,ego,obs,self.carla,self.context)
        self.assertAlmostEqual(r['ego']['longitudinal_speed_mps'],6)
        self.assertTrue(r['checks']['forward_speed_within_planner_bounds'])
        self.assertFalse(r['planner_initial_state_candidate']['ready_for_live_replanning'])
    def test_steering_nonzero_unverified(self):
        ego,obs,snap=self.make_pair(wheel=(5,5))
        r=m.build_observation(snap,ego,obs,self.carla,self.context)
        self.assertIsNone(r['planner_initial_state_candidate']['steer_rad'])
    def test_obstacle_mismatch_flag(self):
        ego,obs,snap=self.make_pair()
        obs.pose.location.x+=1
        r=m.build_observation(snap,ego,obs,self.carla,self.context)
        self.assertFalse(r['checks']['obstacle_center_matches_offline_location'])
    def test_actor_types_enforced(self):
        ego,obs,snap=self.make_pair()
        obs.type_id='vehicle.something'
        with self.assertRaises(ValueError):m.build_observation(snap,ego,obs,self.carla,self.context)
    def test_actor_ids_distinct(self):
        ego,obs,snap=self.make_pair()
        with self.assertRaises(ValueError):m.build_observation(snap,ego,ego,self.carla,self.context)
    def test_observe_existing_is_read_only(self):
        ego,obs,snap=self.make_pair()
        class World:
            def get_actor(self,ident):return {10:ego,11:obs}.get(ident)
            def get_snapshot(self):return snap
        args=obj(ego_id=10,obstacle_id=11)
        r=m.capture_existing(args,self.context,self.carla,World(),{'world_synchronous':True})
        self.assertEqual(r['owned_actor_ids'],[])
        self.assertEqual(r['client_world_ticks'],0)
    def test_mock_spawn_snapshot_and_cleanup(self):
        world=FakeWorld()
        bps=(obj(id='vehicle.tesla.model3'),obj(id='static.prop.constructioncone'))
        args=obj(confirm_spawn=True,ego_z_offset=.5,timeout=1.)
        info={'world_synchronous':False,'existing_vehicle_count':0}
        r=m.spawn_capture(args,self.context,self.carla,world,obj(transform=pose(384.591,-1.993,0)),info,bps)
        self.assertTrue(r['cleanup_ok'])
        self.assertEqual(world.tick_calls,0)
        self.assertTrue(all(a.destroyed for a in world.actors))
        self.assertLess(r['errors_vs_offline']['obstacle_center_position_m'],.01)
        self.assertEqual(r['observation_mode'],'temporary_spawn_passive_snapshot')
        self.assertFalse(r['checks']['forward_speed_within_planner_bounds'])
    def test_mock_spawn_failure_cleans_ego(self):
        world=FakeWorld();world.can_spawn_prop=False
        args=obj(confirm_spawn=True,ego_z_offset=.5,timeout=1.)
        info={'world_synchronous':False,'existing_vehicle_count':0}
        bps=(obj(id='vehicle.tesla.model3'),obj(id='static.prop.constructioncone'))
        with self.assertRaises(RuntimeError):
            m.spawn_capture(args,self.context,self.carla,world,obj(transform=pose(384.591,-1.993,0)),info,bps)
        self.assertTrue(world.actors[0].destroyed)
    def test_no_spawn_in_sync_or_occupied_world(self):
        world=FakeWorld();args=obj(confirm_spawn=True,ego_z_offset=.5,timeout=1.)
        for info in ({'world_synchronous':True,'existing_vehicle_count':0},
                     {'world_synchronous':False,'existing_vehicle_count':1}):
            with self.assertRaises(RuntimeError):
                m.spawn_capture(args,self.context,self.carla,world,obj(transform=pose(0,0,0)),info,
                                (obj(id='vehicle.x'),obj(id='static.prop.x')))
        self.assertEqual(world.actors,[])
    def test_json_serialization_and_overwrite_independent(self):
        ego,obs,snap=self.make_pair()
        report=m.build_observation(snap,ego,obs,self.carla,self.context)
        with tempfile.TemporaryDirectory() as d:
            p=m.save_json(Path(d)/'observation.json',report)
            loaded=json.loads(p.read_text())
            self.assertEqual(loaded['snapshot']['frame'],12)
    def test_offline_first_reference_speed_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ref.csv';p.write_text('t_s,speed_mps\n0,6\n0.1,7\n')
            self.assertEqual(m.first_reference_speed(p),6.)
            p.write_text('t_s,speed_mps\n1,6\n')
            with self.assertRaises(ValueError):m.first_reference_speed(p)
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):m.finite('bad',float('nan'))
    def test_degenerate_route_rejected(self):
        with self.assertRaises(ValueError):m.project_polyline([(0,1,2),(0,1,2)],2,3)

    def test_offline_provenance_integration(self):
        import csv,hashlib
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            route=root/'route.csv'
            route.write_text('x_m,y_m\n384.591,1.993\n334.591,1.993\n')
            safety=root/'safety.csv'; corridor=root/'corridor.csv'
            for p in (safety,corridor):
                p.write_text('t_s,x_m,y_m,yaw_rad,speed_mps\n0,384.591,1.993,3.141592653589793,6\n')
            summary=root/'summary.json'
            settings={'obstacle_station_m':21.,'obstacle_offset_m':-2.4,'obstacle_radius_m':.65,
                      'clearance_margin_m':1.6,'wheelbase_m':2.7,'speed_min_mps':5.,'speed_max_mps':8.,
                      'length_m':4.5,'width_m':1.9,'max_abs_route_offset_m':2.5,
                      'max_abs_heading_error_deg':35.,'max_abs_steering_deg':25.}
            doc={'route_source':str(route),'route_sha256':hashlib.sha256(route.read_bytes()).hexdigest(),
                 'settings':settings,'obstacle_xy_m':[363.591,4.393]}
            summary.write_text(json.dumps(doc))
            ctx=m.offline_context(summary,route,safety,corridor)
            self.assertAlmostEqual(ctx['nominal_speed_mps'],6)
            self.assertAlmostEqual(ctx['plan']['obstacle_model']['commonroad_xy_m'][1],4.393)
            doc['route_sha256']='wrong'
            summary.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):m.offline_context(summary,route,safety,corridor)

    def test_rejects_unexpected_actor_ids_in_offline_mode(self):
        # Argument errors occur before any source file is read or CARLA is connected.
        with self.assertRaises(SystemExit) as e:
            m.main(['--ego-id','10'])
        self.assertEqual(e.exception.code,2)

if __name__=='__main__':unittest.main(verbosity=2)
