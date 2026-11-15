"""Synthetic regression checks for Milestone 4E-A (no CARLA/CommonRoad runtime)."""
from dataclasses import replace
from shapely.geometry import Polygon, box
import unittest

from town01_motion_graph import (
    Settings, build_lattice, initial_state, next_state, select_trajectory,
    synthetic_route, unpack_state,
)
from town01_feasibility_check import Limits
from town01_constraint_graph import (
    VALIDATION_ROAD_IDS, PLANNED_ROUTE_IDS, build_constrained_lattice,
    edge_road_violation, edge_set, experiment, road_footprint_at,
    sampled_edge_road_ok,
)


class TestConstraintAwareGraph(unittest.TestCase):
    def setUp(self):
        self.route = synthetic_route()
        self.cfg = Settings()
        self.limits = Limits(sample_dt_s=0.1)
        self.wide = box(-10, -10, 90, 10)
        self.narrow = box(-4, -1.75, 84, 1.75)
        self.start = initial_state(self.cfg)
        self.forward = next_state(self.start, 0., 0., self.cfg, self.route[-1][0])

    def test_validation_region_includes_predecessor_but_route_does_not(self):
        self.assertEqual(PLANNED_ROUTE_IDS, (38, 40, 58, 60))
        self.assertEqual(VALIDATION_ROAD_IDS, (39, 38, 40, 58, 60))

    def test_footprint_uses_rectangle_not_center(self):
        footprint = road_footprint_at(self.route, 0, 1.0, 6, 0, self.limits)
        self.assertAlmostEqual(footprint.area, 4.5 * 1.9)
        # Center stays inside 3.5m-wide road, but full footprint exceeds it.
        self.assertFalse(self.narrow.covers(footprint))
        self.assertTrue(self.wide.covers(footprint))

    def test_centerline_motion_inside_wide_road(self):
        self.assertTrue(sampled_edge_road_ok(self.start, self.forward, 0, 0,
                           self.cfg, self.route, self.wide, self.limits))
        self.assertIsNone(edge_road_violation(self.start, self.forward, 0, 0,
                           self.cfg, self.route, self.wide, self.limits))

    def test_edge_endpoint_outside_road_is_rejected(self):
        # Road covers initial footprint but not 6m forward travel.
        short_road = box(76.0, -2.0, 84.0, 2.0)
        self.assertTrue(short_road.covers(road_footprint_at(
            self.route, 0, 0, 6, 0, self.limits)))
        self.assertEqual('road_footprint', edge_road_violation(
            self.start, self.forward, 0, 0, self.cfg, self.route, short_road, self.limits))

    def test_mid_edge_invalid_even_if_endpoints_safe(self):
        # Separate drivable strips. The vehicle at s=0 or s=6 is fully
        # inside one strip, but must cross a gap during the transition.
        split_road = box(70, -2, 85, 2).difference(box(76.6, -3, 77.4, 3))
        self.assertTrue(split_road.covers(road_footprint_at(self.route,0,0,6,0,self.limits)))
        self.assertTrue(split_road.covers(road_footprint_at(self.route,6,0,6,0,self.limits)))
        self.assertEqual('road_footprint', edge_road_violation(
            self.start, self.forward, 0, 0, self.cfg, self.route, split_road, self.limits))

    def test_quantization_jump_rejected(self):
        t, si, di, vi, yi = self.forward
        fake_snapped = (t, si + 1, di, vi, yi)
        self.assertEqual('quantization', edge_road_violation(
            self.start, fake_snapped, 0, 0, self.cfg, self.route, self.wide, self.limits))

    def test_bad_start_fails_before_search(self):
        with self.assertRaisesRegex(ValueError, 'Initial ego footprint'):
            build_constrained_lattice(self.route, self.cfg, box(80, -1, 90, 1), self.limits)

    def test_invalid_road_fails(self):
        bowtie = Polygon([(0,0), (4,4), (0,4), (4,0)])
        with self.assertRaisesRegex(ValueError, 'Invalid road polygon'):
            build_constrained_lattice(self.route, self.cfg, bowtie, self.limits)

    def test_unconstrained_road_matches_m4b_exactly(self):
        a_start, a_graph, a_terminal, a_counts, _ = build_lattice(self.route, self.cfg)
        b_start, b_graph, b_terminal, b_counts, _, rejections = build_constrained_lattice(
            self.route, self.cfg, self.wide, self.limits)
        self.assertEqual(a_start, b_start)
        self.assertEqual(a_terminal, b_terminal)
        self.assertEqual(edge_set(a_graph), edge_set(b_graph))
        self.assertEqual(a_counts, b_counts)
        self.assertEqual(rejections.get('road_footprint', 0), 0)
        for p in ('safety_first', 'corridor_first'):
            self.assertEqual(select_trajectory(a_start,a_graph,a_terminal,p,self.cfg),
                             select_trajectory(b_start,b_graph,b_terminal,p,self.cfg))

    def test_narrow_road_prunes_before_selection(self):
        a_start, a_graph, _, a_counts, _ = build_lattice(self.route, self.cfg)
        b_start, b_graph, b_terminal, b_counts, _, rejected = build_constrained_lattice(
            self.route, self.cfg, self.narrow, self.limits)
        self.assertTrue(edge_set(b_graph).issubset(edge_set(a_graph)))
        self.assertLess(sum(x['edges_from_previous'] for x in b_counts),
                        sum(x['edges_from_previous'] for x in a_counts))
        self.assertGreater(rejected.get('road_footprint', 0), 0)
        self.assertEqual(len(b_terminal), b_counts[-1]['states'])

    def test_same_constrained_graph_both_profiles_and_diagnostics(self):
        report, saved, _ = experiment(self.route,self.narrow,self.cfg,self.limits)
        self.assertTrue(report['graph']['one_shared_feasible_graph_for_both_priorities'])
        self.assertGreater(report['graph']['edges'],0)
        self.assertTrue(saved['safety_first']['diagnostic']['road_pass'])
        self.assertTrue(saved['corridor_first']['diagnostic']['road_pass'])
        self.assertTrue(report['baseline_unconstrained_graph']['selection']['safety_first']['changed_by_road_constraint'])
        self.assertLess(report['profiles']['safety_first']['costs']['R5_proxy'],1.9725)

    def test_progress_requirement_applies_to_both_profiles(self):
        report, saved, _ = experiment(self.route, self.narrow, self.cfg, self.limits, min_final_s=40.0)
        self.assertGreaterEqual(saved['safety_first']['rows'][-1]['s_m'],40)
        self.assertGreaterEqual(saved['corridor_first']['rows'][-1]['s_m'],40)
        self.assertEqual(report['minimum_terminal_progress_m'],40)

    def test_no_full_horizon_path_is_reported(self):
        road = box(70,-2,84,2)
        with self.assertRaisesRegex(RuntimeError, 'no reachable states'):
            build_constrained_lattice(self.route,self.cfg,road,self.limits)

    def test_tighter_road_cannot_add_edges(self):
        start, wide_graph, _, _, _, _ = build_constrained_lattice(
            self.route,self.cfg,self.wide,self.limits)
        _, narrow_graph, _, _, _, _ = build_constrained_lattice(
            self.route,self.cfg,self.narrow,self.limits)
        self.assertTrue(edge_set(narrow_graph).issubset(edge_set(wide_graph)))


if __name__ == '__main__':
    unittest.main()
