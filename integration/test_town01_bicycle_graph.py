"""Unit / small integrated smoke tests for AI-authored Milestone 4E-B.

No Town01 XML needed for tests. Run from integration directory:
  python -m unittest -v test_town01_bicycle_graph
"""
from dataclasses import replace
import math
import tempfile
import unittest
from pathlib import Path

from shapely.geometry import box

from town01_bicycle_graph import (
    body_center, Config, Node, Vehicle, build_motion_tree, check_selected, compare_paths,
    initial_vehicle, kinematic_derivative, path_samples, primitive,
    rank, rk4_step, route_projection, select_diverse_candidates,
    select_path, validate_config, wrap,
)
from town01_motion_graph import synthetic_route


class BicycleModelTests(unittest.TestCase):
    def setUp(self):
        self.route=synthetic_route()
        self.road=box(-4,-1.75,84,1.75)
        self.cfg=Config()

    def root(self, v, station=0.):
        return Node(0,0,v,None,None,station,0.,0.,(0.,0.,0.),[])

    def test_initial_body_center_on_route_not_rear_axle(self):
        v=initial_vehicle(self.route,self.cfg)
        self.assertAlmostEqual(body_center(v,self.cfg)[0],80.)
        self.assertAlmostEqual(body_center(v,self.cfg)[1],0.)
        self.assertAlmostEqual(v.x,81.35)

    def test_straight_constant_speed_is_analytic(self):
        q=Vehicle(0.,0.,0.,6.,0.)
        z=rk4_step(q,0.,0.,1.,2.7)
        self.assertAlmostEqual(z.x,6.,places=9)
        self.assertAlmostEqual(z.y,0.,places=9)
        self.assertAlmostEqual(z.yaw,0.,places=9)
        self.assertAlmostEqual(z.v,6.,places=9)

    def test_longitudinal_acceleration_analytic(self):
        q=Vehicle(0,0,0,6,0)
        z=rk4_step(q,1.,0.,1.,2.7)
        self.assertAlmostEqual(z.x,6.5,places=9)
        self.assertAlmostEqual(z.v,7.,places=9)

    def test_steering_rate_integrates_continuous_steering(self):
        q=Vehicle(0,0,0,6,0)
        z=rk4_step(q,0,8,1,2.7)
        self.assertAlmostEqual(math.degrees(z.steer),8.,places=8)
        self.assertGreater(z.y,0)
        self.assertGreater(z.yaw,0)
        self.assertAlmostEqual(z.v,6.)

    def test_yaw_wrap(self):
        self.assertAlmostEqual(wrap(3*math.pi),math.pi)
        self.assertAlmostEqual(wrap(-3*math.pi),-math.pi)

    def test_route_projection_westbound_left_positive(self):
        s,d,heading=route_projection(self.route,74.,-1.)
        self.assertAlmostEqual(s,6.)
        self.assertAlmostEqual(d,1.)
        self.assertAlmostEqual(abs(heading),math.pi)

    def test_validate_rejects_invalid_settings(self):
        with self.assertRaises(ValueError):
            validate_config(replace(self.cfg,beam_width=0))
        with self.assertRaises(ValueError):
            validate_config(replace(self.cfg,integration_dt_s=0))
        with self.assertRaises(ValueError):
            validate_config(replace(self.cfg,speed_initial_mps=12))

    def test_primitive_continuity_zero_steering(self):
        root=self.root(initial_vehicle(self.route,self.cfg))
        result,reason=primitive(root,0.,0.,self.cfg,self.route,self.road,(59.,2.4))
        self.assertIsNone(reason)
        samples,cost,station,offset,head=result
        self.assertEqual(len(samples),10)
        self.assertAlmostEqual(station,6.)
        self.assertAlmostEqual(offset,0.)
        self.assertAlmostEqual(body_center(samples[-1][1],self.cfg)[0],74.)
        self.assertAlmostEqual(cost[1],0.)

    def test_mid_edge_road_crossing_rejected_even_if_endpoints_inside(self):
        route=[(0.,0.,0.),(20.,20.,0.)]
        road=box(-4,-2,20,2).difference(box(3,-2,4,2))
        cfg=replace(self.cfg,dt_s=1.5)
        root=self.root(Vehicle(-1.35,0.,0.,6.,0.))
        # Initial BODY front=2.25 < gap x=3; final rear=6.75 > gap x=4.
        self.assertAlmostEqual(body_center(root.state,cfg)[0],0.)
        self.assertTrue(road.covers(box(-2.25,-.95,2.25,.95)))
        self.assertTrue(road.covers(box(6.75,-.95,11.25,.95)))
        result,reason=primitive(root,0.,0.,cfg,route,road,(100.,0.))
        self.assertIsNone(result)
        self.assertEqual(reason,'road_footprint')

    def test_hard_obstacle_collision_rejected(self):
        route=[(0.,0.,0.),(20.,20.,0.)]
        cfg=replace(self.cfg)
        root=self.root(Vehicle(0.,0.,0.,6.,0.))
        result,reason=primitive(root,0.,0.,cfg,route,box(-4,-4,20,4),(6.,0.))
        self.assertIsNone(result)
        self.assertEqual(reason,'hard_obstacle_collision')

    def test_speed_limit_rejected(self):
        route=[(0.,0.,0.),(20.,20.,0.)]
        root=self.root(Vehicle(0.,0.,0.,8.,0.))
        result,reason=primitive(root,1.,0.,self.cfg,route,box(-4,-4,20,4),(100.,0.))
        self.assertIsNone(result)
        self.assertEqual(reason,'speed_limit')

    def test_steer_limit_rejected(self):
        route=[(0.,0.,0.),(20.,20.,0.)]
        root=self.root(Vehicle(0.,0.,0.,6.,0.))
        result,reason=primitive(root,0.,30.,self.cfg,route,box(-4,-5,20,5),(100.,0.))
        self.assertIsNone(result)
        self.assertEqual(reason,'steering_limit')

    def test_lateral_accel_rejected(self):
        route=[(0.,0.,0.),(30.,30.,0.)]
        root=self.root(Vehicle(0.,0.,0.,8.,math.radians(10)))
        result,reason=primitive(root,0.,0.,self.cfg,route,box(-4,-10,30,10),(100.,0.))
        self.assertIsNone(result)
        self.assertIn(reason,('invalid_source_lateral_accel_limit','lateral_accel_limit'))

    def test_pruning_does_not_snap_vehicle_pose(self):
        n=Node(12,1,Vehicle(73.829,0.083,math.pi-.01,6.002,0.05),0,(0.,4.),
               6.171,0.083,-.01,(1.,2.,3.),[])
        self.assertIs(select_diverse_candidates([n],self.cfg)[0],n)
        self.assertEqual(n.state.x,73.829)

    def test_rulebook_ranking_is_priority_sensitive(self):
        one=Node(1,1,Vehicle(0,0,0,6,0),0,(0,0),30,0,0,(0.1,1.0,0),[])
        two=Node(2,1,Vehicle(0,0,0,6,0),0,(0,0),30,0,0,(1.0,0.,0),[])
        self.assertLess(rank(one,'safety_first',0),rank(two,'safety_first',0))
        self.assertLess(rank(two,'corridor_first',0),rank(one,'corridor_first',0))
        self.assertIsNone(rank(one,'safety_first',40))

    def test_shared_tree_integrated_paths_and_checks(self):
        cfg=replace(self.cfg,horizon_steps=3,beam_width=100)
        nodes,leaves,layers,rej,obs=build_motion_tree(self.route,self.road,cfg)
        self.assertEqual(len(layers),3)
        self.assertGreater(len(nodes),3)
        self.assertGreater(sum(rej.values()),0)
        for profile in ('safety_first','corridor_first'):
            path=select_path(nodes,leaves,profile)
            rows=path_samples(path,cfg,profile,self.route)
            self.assertEqual(len(rows),1+3*10)
            self.assertTrue(check_selected(path,self.road,obs,cfg)['road_sampled_pass'])
            self.assertTrue(check_selected(path,self.road,obs,cfg)['hard_obstacle_sampled_pass'])
            self.assertAlmostEqual(path[1].samples[-1][1].x,path[1].state.x)
            self.assertAlmostEqual(rows[-1]['x_m'],body_center(path[-1].state,cfg)[0])

    def test_compare_paths_requires_same_time_grid(self):
        rows_a=[{'t_s':0,'x_m':0,'y_m':0,'station_m':0,'offset_m':0},
                {'t_s':1,'x_m':1,'y_m':0,'station_m':1,'offset_m':0}]
        rows_b=[dict(r) for r in rows_a]
        self.assertAlmostEqual(compare_paths(rows_a,rows_b)['max_time_aligned_separation_m'],0)
        rows_b[-1]['t_s']=2
        with self.assertRaises(ValueError):
            compare_paths(rows_a,rows_b)


if __name__=='__main__':
    unittest.main()
