"""Milestone 6B-2 offline unit tests; never import or connect to CARLA.

When the earlier project's geometry helper files are absent from a standalone
archive, the tests supply minimal stubs for those import-only dependencies.
The actual town01_bicycle_graph.py (RK4, beam, ranking) is always imported.
"""
from __future__ import annotations
import copy
from dataclasses import asdict
import importlib.util
import math
import sys
import types
import unittest

from shapely.affinity import rotate, translate
from shapely.geometry import Point, box

if importlib.util.find_spec('town01_motion_graph') is None:
    s = types.ModuleType('town01_motion_graph')
    def pose_at(route, station):
        a,b=route[0],route[-1]
        t=station/b[0]
        return a[1]+t*(b[1]-a[1]),a[2]+t*(b[2]-a[2]),math.atan2(b[2]-a[2],b[1]-a[1])
    s.pose_at=pose_at
    s.world_xy=lambda route,station,offset: (pose_at(route,station)[0],pose_at(route,station)[1]+offset)
    s.read_route_csv=lambda *_: None
    s.synthetic_route=lambda: None
    sys.modules['town01_motion_graph']=s
if importlib.util.find_spec('town01_feasibility_check') is None:
    s=types.ModuleType('town01_feasibility_check')
    def vehicle_rectangle(cx,cy,yaw,length,width):
        return translate(rotate(box(-length/2,-width/2,length/2,width/2),
                                math.degrees(yaw),origin=(0,0)),xoff=cx,yoff=cy)
    s.vehicle_rectangle=vehicle_rectangle
    s.load_town01_road=lambda *_: None
    sys.modules['town01_feasibility_check']=s

import town01_bicycle_graph as original
import carla_plan_m6b2 as b


def observation():
    x, y=384.591, 1.993
    theta=math.pi
    ox,oy=363.592,4.405
    delta=.17203
    corners=[(ox-delta,oy-delta),(ox-delta,oy+delta),(ox+delta,oy-delta),(ox+delta,oy+delta)]
    return {'milestone':'6B-1', 'snapshot':{'frame':503572},
            'server':{'map':'Carla/Maps/Town01', 'carla_client_version':'0.9.16',
                      'carla_server_version':'0.9.16'},
            'ego':{'actor_id':198,'type_id':'vehicle.tesla.model3',
                   'body_center_cr_xy_m':[x,y],
                   'longitudinal_speed_mps':0.,'lateral_speed_mps':0.,
                   'velocity_carla_xyz_mps':[0,0,-1.087],
                   'front_wheel_physics_angles_carla_deg':[0.,0.],
                   'bbox':{'local_half_extents_m':[2.3958898,1.081725,.74383]}},
            'obstacle':{'actor_id':199,'type_id':'static.prop.constructioncone',
                        'center_cr_xy_m':[ox,oy],
                        'bbox':{'vertices_commonroad_xy_m':[list(p) for p in corners for _ in (0,1)]},
                        'collision_mesh_verified':False},
            'planner_initial_state_candidate':{
                'rear_axle_x_cr_m':x+1.35,'rear_axle_y_cr_m':y,'yaw_cr_rad':theta,
                'forward_speed_mps':0.,'steer_rad':0.},
            'observation_mode':'temporary_spawn_passive_snapshot', 'cleanup_ok':True}


class Milestone6B2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.route=[(0.,384.591,1.993),(60.,324.591,1.993)]
        cls.road=box(310.,-2.,395.,7.)
        cls.summary={'settings':asdict(original.Config()), 'experiment':'Milestone 4E-B'}
        cls.summary['settings']['beam_width']=25

    def config(self, **options):
        return b.configuration_from_summary(self.summary,**options)

    def test_speed_zero_enabled_without_modifying_previous_planner(self):
        cfg=self.config()
        self.assertEqual(cfg.speed_min_mps,0)
        self.assertEqual(original.Config().speed_min_mps,5)

    def test_root_is_observed_rear_axle_not_route_start_constructor(self):
        obs=observation();cfg=self.config()
        q,issues=b.observed_vehicle(obs,cfg)
        self.assertEqual(q.v,0.)
        self.assertAlmostEqual(q.x,385.941)
        self.assertAlmostEqual(original.body_center(q,cfg)[0],384.591)
        self.assertTrue(any('vertically settled' in s for s in issues))

    def test_real_ego_dimensions_not_older_4p5x1p9(self):
        cfg,raw,hard=b.measured_geometry(observation(),self.config())
        self.assertAlmostEqual(cfg.length_m,4.7917796)
        self.assertAlmostEqual(cfg.width_m,2.16345)
        self.assertGreater(hard.area,raw.area)
        self.assertAlmostEqual(raw.area,(2*.17203)**2,places=5)

    def test_unstable_frame_blocks_default_run_and_does_not_fabricate_speed(self):
        report,paths=b.run_offline(observation(),self.summary,self.route,self.road)
        self.assertEqual(report['status'],'blocked_unstable_observation')
        self.assertIsNone(paths)
        self.assertEqual(report['start_state']['v'],0.)
        self.assertFalse(report['ready_for_live_control'])

    def test_nonzero_steering_rejected_until_calibration(self):
        obs=observation();obs['ego']['front_wheel_physics_angles_carla_deg']=[4.,4.]
        with self.assertRaisesRegex(ValueError,'near-zero'):
            b.observed_vehicle(obs,self.config())

    def test_candidate_speed_must_match_observed_speed(self):
        obs=observation();obs['planner_initial_state_candidate']['forward_speed_mps']=6.
        with self.assertRaisesRegex(ValueError,'differs'):
            b.observed_vehicle(obs,self.config())

    def test_initial_state_collision_rejected(self):
        obs=observation()
        cx,cy=384.591,1.993
        pts=[(cx-.2,cy-.2),(cx-.2,cy+.2),(cx+.2,cy-.2),(cx+.2,cy+.2)]
        obs['obstacle']['center_cr_xy_m']=[cx,cy]
        obs['obstacle']['bbox']['vertices_commonroad_xy_m']=[list(p) for p in pts for _ in (0,1)]
        cfg,raw,hard=b.measured_geometry(obs,self.config())
        q,_=b.observed_vehicle(obs,cfg)
        with self.assertRaisesRegex(ValueError,'hard_obstacle_collision'):
            b.build_observed_motion_tree(self.route,self.road,cfg,q,hard)

    def test_zero_speed_short_primitive_is_possible(self):
        obs=observation()
        cfg,raw,hard=b.measured_geometry(obs,self.config())
        q,_=b.observed_vehicle(obs,cfg)
        nodes,terminals,layers,rejects=b.build_observed_motion_tree(
            self.route,self.road,b.configuration_from_summary(self.summary,horizon_steps=1,beam_width=20),q,hard)
        self.assertTrue(any(node.state.v>0 for node in terminals))
        self.assertAlmostEqual(nodes[0].state.v,0.)

    def test_offline_override_generates_search_but_not_live_ready(self):
        report,selected=b.run_offline(observation(),self.summary,self.route,self.road,
                                      horizon_steps=2,beam_width=25,min_progress_m=.2,
                                      allow_unstable_offline=True)
        self.assertEqual(report['status'],'offline_plan_generated_unstable_input')
        self.assertFalse(report['ready_for_live_control'])
        self.assertTrue(report['offline_unstable_override_used'])
        self.assertEqual(report['start_state']['v'],0)
        self.assertEqual(set(selected),{'safety_first','corridor_first'})
        for profile, data in selected.items():
            self.assertTrue(data['diagnostic']['sampled_road_pass'])
            self.assertTrue(data['diagnostic']['sampled_hard_obstacle_pass'])

    def test_observed_lateral_motion_rejected(self):
        obs=observation();obs['ego']['lateral_speed_mps']=1.0
        with self.assertRaisesRegex(ValueError,'lateral'):
            b.observed_vehicle(obs,self.config())

    def test_bad_obstacle_box_rejected(self):
        obs=observation();obs['obstacle']['bbox']['vertices_commonroad_xy_m']=[[1,1]]*8
        with self.assertRaisesRegex(ValueError,'degenerate'):
            b.measured_geometry(obs,self.config())

    def test_speed_range_prevents_negative_reverse_and_overspeed(self):
        obs=observation()
        for value in (-1, 9):
            o=copy.deepcopy(obs)
            o['planner_initial_state_candidate']['forward_speed_mps']=value
            with self.assertRaises(ValueError):
                b.observed_vehicle(o,self.config())

    def test_stable_input_can_plan_without_override_but_stays_offline(self):
        obs=observation()
        obs['ego']['velocity_carla_xyz_mps'][2]=0.
        report,paths=b.run_offline(obs,self.summary,self.route,self.road,
                                   horizon_steps=2,beam_width=25,min_progress_m=.2)
        self.assertEqual(report['status'],'offline_plan_generated')
        self.assertFalse(report['ready_for_live_control'])
        self.assertTrue(paths)

    def test_infeasible_observed_footprint_returns_failure_report(self):
        obs=observation()
        obs['ego']['velocity_carla_xyz_mps'][2]=0.
        skinny_road=box(310.,1.0,395.,2.5)
        report,paths=b.run_offline(obs,self.summary,self.route,skinny_road,
                                   horizon_steps=2,beam_width=20,min_progress_m=.2)
        self.assertEqual(report['status'],'offline_planning_failed')
        self.assertIn('road_footprint',report['explanation'])
        self.assertIsNone(paths)

    def test_cross_experiment_obstacle_provenance_rejected(self):
        obs=observation()
        obs['offline_obstacle']={'commonroad_xy_m':[363.592,4.405]}
        summary=copy.deepcopy(self.summary)
        summary['obstacle_xy_m']=[363.592,4.405]
        b.verify_observation_scenario(obs,summary)
        summary['obstacle_xy_m']=[111.,222.]
        with self.assertRaisesRegex(ValueError,'mixed experiments'):
            b.verify_observation_scenario(obs,summary)

    def test_preserves_priority_order_for_terminal_candidates(self):
        a=original.Node(1,1,original.Vehicle(0,0,0,0,0),0,None,1,0,0,(0,1,0),[])
        z=original.Node(2,1,original.Vehicle(0,0,0,0,0),0,None,1,0,0,(1,0,0),[])
        self.assertLess(original.rank(a,'safety_first',0),original.rank(z,'safety_first',0))
        self.assertLess(original.rank(z,'corridor_first',0),original.rank(a,'corridor_first',0))


    def verified_settled_observation(self, speed=-0.00003):
        obs = observation()
        obs['ego']['velocity_carla_xyz_mps'] = [-speed, 0.0, 0.001]
        obs['ego']['longitudinal_speed_mps'] = speed
        obs['planner_initial_state_candidate']['forward_speed_mps'] = speed
        obs['stability_capture_mode'] = 'passive_multi_frame'
        obs['stability'] = {
            'criteria': {'max_ego_horizontal_speed_mps': 0.05,
                         'max_ego_vertical_speed_mps': 0.05},
            'latest': {'accepted': True,
                       'ego_horizontal_speed_mps': abs(speed),
                       'ego_vertical_speed_mps': 0.001}}
        return obs

    def test_verified_settled_negative_residual_is_zero_model_speed(self):
        obs = self.verified_settled_observation()
        q, issues = b.observed_vehicle(obs, self.config())
        self.assertEqual(q.v, 0.0)
        self.assertEqual(obs['ego']['longitudinal_speed_mps'], -0.00003)
        self.assertFalse(any('vertically settled' in issue for issue in issues))

    def test_verified_settled_negative_residual_runs_without_override(self):
        obs = self.verified_settled_observation()
        report, paths = b.run_offline(obs, self.summary, self.route, self.road,
                                      horizon_steps=2, beam_width=25, min_progress_m=0.2)
        self.assertEqual(report['status'], 'offline_plan_generated')
        self.assertTrue(report['stationary_negative_speed_normalized_to_zero'])
        self.assertEqual(report['raw_observed_forward_speed_mps'], -0.00003)
        self.assertEqual(report['start_state']['v'], 0.0)
        self.assertTrue(paths)
        self.assertFalse(report['ready_for_live_control'])

    def test_negative_residual_without_settle_evidence_rejected(self):
        obs = observation()
        obs['ego']['longitudinal_speed_mps'] = -0.00003
        obs['planner_initial_state_candidate']['forward_speed_mps'] = -0.00003
        with self.assertRaisesRegex(ValueError, 'outside the supported'):
            b.observed_vehicle(obs, self.config())

    def test_meaningful_reverse_motion_still_rejected(self):
        obs = self.verified_settled_observation(-0.08)
        with self.assertRaisesRegex(ValueError, 'outside the supported'):
            b.observed_vehicle(obs, self.config())

    def test_failed_stability_evidence_rejects_negative_residual(self):
        obs = self.verified_settled_observation(-0.00003)
        obs['stability']['latest']['accepted'] = False
        with self.assertRaisesRegex(ValueError, 'outside the supported'):
            b.observed_vehicle(obs, self.config())

    def test_raw_measurement_candidate_mismatch_still_rejected(self):
        obs = self.verified_settled_observation(-0.00003)
        obs['planner_initial_state_candidate']['forward_speed_mps'] = -0.03
        with self.assertRaisesRegex(ValueError, 'differs from CARLA'):
            b.observed_vehicle(obs, self.config())

if __name__=='__main__':
    unittest.main(verbosity=2)
