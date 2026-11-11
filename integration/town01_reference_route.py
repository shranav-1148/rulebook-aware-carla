"""Town01 Milestone 3B: plan a reference path, without running RPPSolver.


Uses installed CommonRoad-IO and commonroad-route-planner public APIs.
Reference API: https://commonroad.in.tum.de/tutorials/5_tutorial_route_planner.html
No upstream repository files are modified.

Input:  integration/maps/Town01.route_38_40_58_60.xml
Output: integration/output/town01_route/{reference_path.csv,summary.json,reference_path.png}
"""

import csv
import json
from pathlib import Path

import numpy as np

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.geometry.shape import Circle, Polygon, ShapeGroup
from shapely.geometry import Point as ShapelyPoint, Polygon as ShapelyPolygon
from commonroad_route_planner.route_planner import RoutePlanner


BASE = Path(__file__).resolve().parent
SCENARIO_FILE = BASE / "maps" / "Town01.route_38_40_58_60.xml"
OUTPUT_DIR = BASE / "output" / "town01_route"
EXPECTED_CHAIN = [38, 40, 58, 60]


def path_length(points):
    """Sum distances between consecutive reference-path points, in meters."""
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def main():
    # 1. Load the map and synthetic goal we already verified in Milestone 3A.
    scenario, problem_set = CommonRoadFileReader(str(SCENARIO_FILE)).open()
    problems = list(problem_set.planning_problem_dict.values())
    if len(problems) != 1:
        raise RuntimeError(f"Expected one planning problem; found {len(problems)}")
    problem = problems[0]

    # 2. This matches the Scenario-based RoutePlanner API used in Lesson 7.
    planner = RoutePlanner(scenario, problem)
    candidate_holder = planner.plan_routes()

    # API of the previously used commonroad-route-planner 2024.x generation.
    candidates, candidate_count = candidate_holder.retrieve_all_routes()
    if candidate_count == 0 or not candidates:
        raise RuntimeError("RoutePlanner found no route from the initial state to the goal")

    route = candidate_holder.retrieve_first_route()
    points = np.asarray(route.reference_path, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise RuntimeError(f"Unexpected reference-path array shape: {points.shape}")
    if not np.isfinite(points).all():
        raise RuntimeError("Reference path contains NaN or infinity")

    # Old Route objects commonly use 'list_ids_lanelets'; other releases
    # may use 'lanelet_ids'. Do not guess if neither is exposed.
    lanelet_ids = getattr(route, "list_ids_lanelets", None)
    if lanelet_ids is None:
        lanelet_ids = getattr(route, "lanelet_ids", None)
    lanelet_ids = None if lanelet_ids is None else [int(i) for i in lanelet_ids]

    start = np.asarray(problem.initial_state.position, dtype=float)
    goal_shape = problem.goal.state_list[0].position
    if isinstance(goal_shape, ShapeGroup):
        if len(goal_shape.shapes) != 1:
            raise ValueError(
                f"Expected one goal shape in this experiment; "
                f"found {len(goal_shape.shapes)}"
            )
        goal_shape = goal_shape.shapes[0]

    # The XML writer may serialize lanelet goal metadata as a lanelet reference,
    # which loads as a Polygon. A polygon must NOT be treated as a radius-1.5 circle.
    polygon_vertices = None
    goal_radius = None
    if isinstance(goal_shape, Circle):
        goal_center = np.asarray(goal_shape.center, dtype=float)
        goal_radius = float(goal_shape.radius)
        distances_to_goal_region = np.maximum(
            0.0, np.linalg.norm(points - goal_center, axis=1) - goal_radius
        )
    elif isinstance(goal_shape, Polygon):
        polygon_vertices = np.asarray(goal_shape.vertices, dtype=float)
        actual_polygon = ShapelyPolygon(polygon_vertices)
        if not actual_polygon.is_valid or actual_polygon.is_empty:
            raise ValueError("The reloaded polygon goal is invalid")
        goal_center = np.asarray(actual_polygon.centroid.coords[0], dtype=float)
        distances_to_goal_region = np.asarray(
            [actual_polygon.distance(ShapelyPoint(float(x), float(y)))
             for x, y in points],
            dtype=float,
        )
    else:
        raise TypeError(
            f"Unsupported reloaded goal shape: {type(goal_shape).__name__}"
        )

    start_distance = float(np.min(np.linalg.norm(points - start, axis=1)))
    goal_distance = float(np.min(np.linalg.norm(points - goal_center, axis=1)))
    goal_region_distance = float(np.min(distances_to_goal_region))
    length = path_length(points)

    # 3. Save numerical output that later motion-planning experiments can read.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_file = OUTPUT_DIR / "reference_path.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["point_index", "x_m", "y_m"])
        for index, (x, y) in enumerate(points):
            writer.writerow([index, float(x), float(y)])

    summary = {
        "source_xml": str(SCENARIO_FILE),
        "route_candidates": int(candidate_count),
        "selected_lanelet_ids": lanelet_ids,
        "previously_verified_successor_chain": EXPECTED_CHAIN,
        "reference_path_points": int(len(points)),
        "reference_path_length_m": length,
        "start_position_m": start.tolist(),
        "goal_shape_type": type(goal_shape).__name__,
        "goal_representative_center_m": goal_center.tolist(),
        "goal_radius_m_if_circle": goal_radius,
        "goal_polygon_vertex_count": None if polygon_vertices is None else int(len(polygon_vertices)),
        "closest_reference_point_to_start_m": start_distance,
        "closest_reference_point_to_goal_representative_center_m": goal_distance,
        "closest_reference_point_to_actual_goal_region_m": goal_region_distance,
        "note": "Geometric route only; not a timed or collision-checked trajectory",
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    # 4. Save a static plot. Use Agg so WSL needs no GUI display.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.5))
    network = scenario.lanelet_network
    to_draw = lanelet_ids if lanelet_ids else EXPECTED_CHAIN
    for lanelet_id in to_draw:
        lanelet = network.find_lanelet_by_id(int(lanelet_id))
        line = np.asarray(lanelet.center_vertices, dtype=float)
        ax.plot(line[:, 0], line[:, 1], color="0.7", linewidth=4, alpha=0.8)
    ax.plot(points[:, 0], points[:, 1], linewidth=2, label="Generated reference path")
    ax.scatter([start[0]], [start[1]], marker="o", s=65, label="Initial state", zorder=5)
    ax.scatter([goal_center[0]], [goal_center[1]], marker="x", s=75,
               label="Goal representative center", zorder=5)
    if polygon_vertices is not None:
        ax.add_patch(plt.Polygon(polygon_vertices, closed=True, fill=False,
                                 linestyle="--", edgecolor="tab:orange"))
    else:
        ax.add_patch(plt.Circle(goal_center, goal_radius, fill=False,
                                linestyle="--", edgecolor="tab:orange"))
    ax.set_xlabel("CommonRoad X (m)")
    ax.set_ylabel("CommonRoad Y (m)")
    ax.set_title("Town01 — route and reference path (not a trajectory)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plot_file = OUTPUT_DIR / "reference_path.png"
    fig.savefig(plot_file, dpi=160)
    plt.close(fig)

    print("\n===== TOWN01 REFERENCE ROUTE =====")
    print("Route candidates:", candidate_count)
    print("Selected lanelet IDs:", lanelet_ids if lanelet_ids else "not exposed by this API")
    print("Reference path points:", len(points))
    print(f"Reference path length: {length:.2f} m")
    print(f"Closest path point to start: {start_distance:.3f} m")
    print("Reloaded goal shape:", type(goal_shape).__name__)
    print(f"Closest path point to goal representative center: {goal_distance:.3f} m")
    print(f"Closest sampled path point to actual goal region: {goal_region_distance:.3f} m")
    print("CSV:", csv_file)
    print("Summary:", OUTPUT_DIR / "summary.json")
    print("Plot:", plot_file)
    if goal_region_distance > 1e-6:
        print("NOTE: No sampled reference-path point is inside the reloaded goal "
              "region; inspect path discretization and goal geometry.")
    if polygon_vertices is not None:
        print("NOTE: The saved XML has a polygon goal (likely the entire goal lanelet), "
              "not the original 1.5 m circle. See milestone notes.")
    print("SUCCESS: Reference route generated. Rulebook motion planning not run.")


if __name__ == "__main__":
    main()
