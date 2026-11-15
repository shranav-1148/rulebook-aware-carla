"""Milestone 4D tests. No CommonRoad or CARLA needed for these synthetic cases."""
import csv
import math
from pathlib import Path
import tempfile
import unittest

from shapely.geometry import Polygon
from town01_motion_graph import synthetic_route
from town01_feasibility_check import (
    Limits, check_dynamics, check_road, inferred_yaw, lanelet_polygon,
    read_motion_csv, reconstruct_dense, road_from_lanelets,
    validate_trajectory, vehicle_rectangle, verify_limits, wrap_angle,
)


class FakeLanelet:
    def __init__(self, idx, start_x, end_x, y0, y1):
        self.lanelet_id = idx
        self.left_vertices = [(start_x, y1), (end_x, y1)]
        self.right_vertices = [(start_x, y0), (end_x, y0)]


def straight_rows(s0=0, s1=6, d0=0, d1=0, v=6, vy=0, ax=0, ay=0):
    return [
        {'t_s':0.,'s_m':float(s0),'d_m':float(d0),'v_mps':float(v),
         'lateral_v_mps':float(vy),'x_m':80.-s0,'y_m':0.,'ax_mps2':float(ax),'ay_mps2':float(ay)},
        {'t_s':1.,'s_m':float(s1),'d_m':float(d1),'v_mps':float(v)+ax,
         'lateral_v_mps':float(vy)+ay,'x_m':80.-s1,'y_m':0.,'ax_mps2':None,'ay_mps2':None}
    ]


class TestFeasibility(unittest.TestCase):
    def setUp(self):
        self.route=synthetic_route()
        self.limits=Limits()
        self.road=Polygon([(-4,-1.75),(84,-1.75),(84,1.75),(-4,1.75)])

    def test_rectangle_aligned_has_expected_size(self):
        rect=vehicle_rectangle(0,0,0,4.5,1.9)
        self.assertAlmostEqual(rect.area,8.55)
        self.assertEqual(tuple(round(v,6) for v in rect.bounds),(-2.25,-0.95,2.25,0.95))

    def test_rotated_rectangle(self):
        rect=vehicle_rectangle(0,0,math.pi/2,4.0,2.0)
        self.assertEqual(tuple(round(v,6) for v in rect.bounds),(-1.0,-2.0,1.0,2.0))

    def test_lanelet_to_polygon(self):
        l=FakeLanelet(38,0,10,-2,2)
        self.assertAlmostEqual(lanelet_polygon(l.left_vertices,l.right_vertices).area,40)

    def test_lanelet_union_across_successor_seam(self):
        road=road_from_lanelets([FakeLanelet(38,0,10,-2,2),FakeLanelet(40,10,20,-2,2)],(38,40))
        self.assertTrue(road.covers(vehicle_rectangle(10,0,0,4,2)))

    def test_reject_bad_lanelet(self):
        with self.assertRaises(ValueError):
            lanelet_polygon([(0,0)],[(0,1),(1,1)])
        with self.assertRaises(ValueError):
            road_from_lanelets([FakeLanelet(38,0,10,-2,2)],(38,40))

    def test_inferred_heading_for_straight_route(self):
        self.assertAlmostEqual(wrap_angle(inferred_yaw(self.route,10,6,0)-math.pi),0,places=8)
        self.assertAlmostEqual(wrap_angle(inferred_yaw(self.route,10,6,1)-math.pi),math.atan2(1,6),places=8)

    def test_dense_one_second_11_samples(self):
        dense,seams=reconstruct_dense(straight_rows(),self.route,0.1)
        self.assertEqual(len(dense),11)
        self.assertEqual(len(seams),1)
        self.assertAlmostEqual(dense[5]['s_m'],3.0)
        self.assertAlmostEqual(seams[0]['position_jump_in_route_frame_m'],0)

    def test_quantization_seam_detected(self):
        dense,seams=reconstruct_dense(straight_rows(s1=6.25),self.route,0.1)
        self.assertAlmostEqual(seams[0]['position_jump_in_route_frame_m'],0.25)
        result=check_dynamics(dense,seams,self.limits)
        self.assertFalse(result['quantization_pass'])

    def test_centerline_car_inside_road(self):
        dense,_=reconstruct_dense(straight_rows(),self.route,0.1)
        result=check_road(dense,self.road,self.limits)
        self.assertTrue(result['road_pass'])
        self.assertEqual(result['road_failure_samples'],0)

    def test_center_alone_not_sufficient_for_footprint(self):
        dense,_=reconstruct_dense(straight_rows(d0=1.1,d1=1.1),self.route,0.1)
        result=check_road(dense,self.road,self.limits)
        self.assertFalse(result['road_pass'])
        self.assertGreater(result['max_footprint_outside_area_m2'],0)

    def test_straight_motion_has_zero_steering_demand(self):
        dense,seams=reconstruct_dense(straight_rows(),self.route,0.1)
        result=check_dynamics(dense,seams,self.limits)
        self.assertTrue(result['bicycle_demand_pass'])
        self.assertAlmostEqual(result['max_required_steering_deg'],0)

    def test_sharp_yaw_change_fails_steering_bound(self):
        samples=[{'t_s':0.,'yaw_rad':0.,'v_approx_mps':5.,'x_m':0.,'y_m':0.},
                 {'t_s':0.1,'yaw_rad':math.pi/2,'v_approx_mps':5.,'x_m':0.5,'y_m':0.}]
        result=check_dynamics(samples,[],self.limits)
        self.assertFalse(result['bicycle_demand_pass'])
        self.assertGreater(result['max_required_steering_deg'],self.limits.max_steering_deg)

    def test_validate_all_checks_for_easy_case(self):
        result,dense,seams=validate_trajectory(straight_rows(),self.route,self.road,self.limits)
        self.assertTrue(result['all_diagnostic_thresholds_pass'])
        self.assertEqual(len(dense),11)

    def test_validate_limits(self):
        from dataclasses import replace
        with self.assertRaises(ValueError):
            verify_limits(replace(self.limits,wheelbase_m=-1))
        with self.assertRaises(ValueError):
            verify_limits(replace(self.limits,max_steering_deg=91))

    def test_read_motion_csv_parses_missing_last_action(self):
        rows=straight_rows()
        with tempfile.TemporaryDirectory() as folder:
            filepath=Path(folder)/'trajectory.csv'
            with filepath.open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]))
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: '' if v is None else v for k,v in row.items()})
            loaded=read_motion_csv(filepath)
            self.assertEqual(len(loaded),2)
            self.assertIsNone(loaded[-1]['ax_mps2'])


if __name__=='__main__':
    unittest.main()
