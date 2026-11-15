"""Synthetic unit tests for AI-authored Milestone 4B graph prototype.

No CommonRoad or CARLA installation is needed.
"""
import csv
from dataclasses import replace
from pathlib import Path
import math
import tempfile
import unittest

from town01_motion_graph import (
    Settings, read_route_csv, pose_at, world_xy, quantize,
    initial_state, unpack_state, next_state, edge_geometry_and_cost,
    synthetic_route, build_lattice, select_trajectory, path_rows, divergence,
)


class TestMotionGraph(unittest.TestCase):
    def setUp(self):
        self.cfg = Settings()
        self.route = synthetic_route()

    def test_read_waypoint_file_and_deduplicate_lanelet_connection(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "waypoints.csv"
            with p.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["waypoint_id", "lanelet_id", "x_m", "y_m"])
                writer.writeheader()
                for i, x in enumerate((30, 25, 25, 20)):
                    writer.writerow({"waypoint_id": i, "lanelet_id": 38, "x_m": x, "y_m": 0})
            result = read_route_csv(p)
            self.assertEqual(len(result), 3)
            self.assertAlmostEqual(result[-1][0], 10.0)

    def test_geometry_interpolates_and_offsets_to_left(self):
        x, y, yaw = pose_at(self.route, 5.0)
        self.assertAlmostEqual(x, 75)
        self.assertAlmostEqual(y, 0)
        self.assertAlmostEqual(yaw, math.pi)
        self.assertAlmostEqual(world_xy(self.route, 5, 1.0)[1], -1.0)

    def test_initial_motion_state(self):
        self.assertEqual(unpack_state(initial_state(self.cfg), self.cfg), (0, 0, 0, 6, 0))

    def test_motion_kinematics_and_lateral_limit(self):
        start = initial_state(self.cfg)
        nxt = next_state(start, -1.0, 0.5, self.cfg, self.route[-1][0])
        self.assertIsNotNone(nxt)
        self.assertEqual(unpack_state(nxt, self.cfg), (1, 5.5, 0.25, 5, 0.5))
        self.assertIsNone(next_state(start, -1.0, 1.5, self.cfg, self.route[-1][0]))

    def test_collision_rejects_centered_obstacle(self):
        start = initial_state(self.cfg)
        nxt = next_state(start, 0.0, 0.0, self.cfg, self.route[-1][0])
        obstacle = world_xy(self.route, 5.0, 0.0)
        self.assertIsNone(edge_geometry_and_cost(start, nxt, 0, 0, self.cfg, self.route, obstacle))

    def test_safety_cost_decreases_with_greater_clearance(self):
        start = initial_state(self.cfg)
        n0 = next_state(start, 0, 0, self.cfg, self.route[-1][0])
        n1 = next_state(start, 0, 0.5, self.cfg, self.route[-1][0])
        obstacle = world_xy(self.route, 5, -2.4)
        c0 = edge_geometry_and_cost(start, n0, 0, 0, self.cfg, self.route, obstacle)
        c1 = edge_geometry_and_cost(start, n1, 0, 0.5, self.cfg, self.route, obstacle)
        self.assertIsNotNone(c0)
        self.assertIsNotNone(c1)
        self.assertLess(c1[0], c0[0])
        self.assertGreater(c1[1], c0[1])

    def test_same_graph_yields_different_priorities(self):
        start, graph, ends, counts, obstacle = build_lattice(self.route, self.cfg)
        self.assertEqual(len(counts), self.cfg.horizon)
        pa, aa, ca = select_trajectory(start, graph, ends, "safety_first", self.cfg)
        pb, ab, cb = select_trajectory(start, graph, ends, "corridor_first", self.cfg)
        self.assertEqual(len(pa), self.cfg.horizon + 1)
        self.assertEqual(len(pb), self.cfg.horizon + 1)
        self.assertAlmostEqual(ca[0], 0.0, places=8)
        self.assertAlmostEqual(cb[1], 0.0, places=8)
        self.assertLess(ca[0], cb[0])
        self.assertGreater(ca[1], cb[1])
        ra = path_rows(pa, aa, ca, "safety_first", self.cfg, self.route)
        rb = path_rows(pb, ab, cb, "corridor_first", self.cfg, self.route)
        self.assertGreater(divergence(ra, rb)["max_lateral_offset_difference_m"], 0.0)

    def test_no_conflict_can_yield_same_trajectory(self):
        cfg = replace(self.cfg, obstacle_station=65.0)
        start, graph, ends, counts, obstacle = build_lattice(self.route, cfg)
        pa, aa, ca = select_trajectory(start, graph, ends, "safety_first", cfg)
        pb, ab, cb = select_trajectory(start, graph, ends, "corridor_first", cfg)
        self.assertEqual(pa, pb)
        self.assertEqual(ca, cb)

    def test_unusable_input_is_rejected(self):
        with self.assertRaises(ValueError):
            build_lattice(self.route, replace(self.cfg, dt=0))
        with self.assertRaises(ValueError):
            build_lattice(self.route, replace(self.cfg, obstacle_station=500))


if __name__ == "__main__":
    unittest.main()
