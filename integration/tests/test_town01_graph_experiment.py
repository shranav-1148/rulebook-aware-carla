"""Pure-Python tests; no CARLA or CommonRoad import required."""
import math
import unittest
from types import SimpleNamespace

from town01_graph_experiment import (
    build_graph, successor_adjacency, validate_chain, bfs_route,
    distance_weighted_route, sample_centerline, make_route_waypoint_graph,
)


def fake_lanelet(i, successors, points, left=None, right=None, same_left=False, same_right=False):
    return SimpleNamespace(lanelet_id=i, successor=successors,
                           center_vertices=points, adj_left=left, adj_right=right,
                           adj_left_same_direction=same_left,
                           adj_right_same_direction=same_right)


class GraphExperimentTest(unittest.TestCase):
    def setUp(self):
        self.lanelets = [
            fake_lanelet(38, [40, 41], [(30, 0), (20, 0)]),
            fake_lanelet(40, [58], [(20, 0), (10, 0)], right=41, same_right=True),
            fake_lanelet(41, [60], [(20, -4), (10, -4)], left=40, same_left=True),
            fake_lanelet(58, [60], [(10, 0), (0, 0)]),
            fake_lanelet(60, [], [(0, 0), (-10, 0)]),
        ]
        self.nodes, self.edges, self.missing = build_graph(self.lanelets)
        self.adjacency = successor_adjacency(self.nodes, self.edges)

    def test_successor_graph(self):
        self.assertEqual(self.adjacency[38], [40, 41])
        self.assertTrue(validate_chain(self.adjacency, [38, 40, 58, 60]))
        self.assertFalse(validate_chain(self.adjacency, [38, 58, 60]))

    def test_lateral_edges_separated_from_successors(self):
        candidates = [e for e in self.edges if e["type"] == "lateral_candidate"]
        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(e["lane_change_legality_verified"] is False for e in candidates))
        self.assertNotIn(41, self.adjacency[40])

    def test_bfs_dijkstra(self):
        self.assertEqual(bfs_route(self.adjacency, 38, 60), [38, 41, 60])
        path, cost = distance_weighted_route(self.adjacency, self.nodes, 38, 60)
        self.assertEqual(path, [38, 41, 60])
        self.assertAlmostEqual(cost, 20.0)

    def test_missing_reference(self):
        lanelets = [fake_lanelet(2, [99], [(0, 0), (1, 0)])]
        n, e, missing = build_graph(lanelets)
        self.assertEqual(len(n), 1)
        self.assertEqual(len(e), 0)
        self.assertEqual(missing[0]["target"], 99)

    def test_sample_centerline(self):
        points = sample_centerline([(0, 0), (10, 0)], 4)
        self.assertEqual([p["s_m"] for p in points], [0, 4, 8, 10])
        self.assertAlmostEqual(points[-1]["x_m"], 10.0)
        self.assertAlmostEqual(points[1]["heading_rad"], 0.0)

    def test_route_waypoint_graph(self):
        nodes = {i: self.nodes[i] for i in (38, 40)}
        points, edges, gaps = make_route_waypoint_graph(nodes, [38, 40], 5)
        self.assertEqual(len(points), 6)
        self.assertEqual(len(edges), 5)
        self.assertEqual(gaps[0]["gap_m"], 0)
        self.assertEqual(sum(e["type"] == "successor_connector" for e in edges), 1)

    def test_reverse_heading(self):
        points = sample_centerline([(10, 0), (0, 0)], 5)
        self.assertAlmostEqual(points[0]["heading_rad"], math.pi)

    def test_reject_invalid_spacing(self):
        with self.assertRaises(ValueError):
            sample_centerline([(0, 0), (1, 0)], 0)


if __name__ == "__main__":
    unittest.main()
