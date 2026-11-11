
"""Build a synthetic planning problem on converted CARLA Town01.


Uses public CommonRoad-IO APIs; does not modify upstream repositories.

Input:  maps/Town01.rpp_compat.xml
Output: maps/Town01.route_38_40_58_60.xml
"""

import math
from pathlib import Path

import numpy as np

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import (
    CommonRoadFileWriter,
    OverwriteExistingFile,
)
from commonroad.common.util import Interval
from commonroad.geometry.shape import Circle
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import (
    PlanningProblem,
    PlanningProblemSet,
)
from commonroad.scenario.scenario import Tag


from commonroad.scenario.state import InitialState, CustomState


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "maps" / "Town01.rpp_compat.xml"
OUTPUT = ROOT / "maps" / "Town01.route_38_40_58_60.xml"

ROUTE = [38, 40, 58, 60]
INITIAL_SPEED = 8.0


def midpoint_pose(lanelet):
    """Find a position and heading halfway along a lanelet centerline."""
    vertices = np.asarray(lanelet.center_vertices, dtype=float)

    segments = np.diff(vertices, axis=0)
    lengths = np.linalg.norm(segments, axis=1)

    total_length = float(np.sum(lengths))
    if total_length <= 0:
        raise ValueError("Lanelet centerline has zero length")

    halfway = total_length / 2.0
    cumulative = np.cumsum(lengths)

    index = int(np.searchsorted(cumulative, halfway))
    index = min(index, len(lengths) - 1)

    if lengths[index] <= 1e-10:
        raise ValueError("Degenerate centerline segment at midpoint")

    previous_length = float(cumulative[index - 1]) if index > 0 else 0.0
    fraction = (halfway - previous_length) / lengths[index]

    position = vertices[index] + fraction * segments[index]
    heading = math.atan2(segments[index, 1], segments[index, 0])

    return position, heading


def main():
    scenario, _ = CommonRoadFileReader(str(INPUT)).open()
    network = scenario.lanelet_network

    lanelets = [network.find_lanelet_by_id(i) for i in ROUTE]

    # Verify that the intended route is connected.
    for current, following in zip(lanelets, lanelets[1:]):
        if following.lanelet_id not in current.successor:
            raise ValueError(
                f"No successor connection: "
                f"{current.lanelet_id} -> {following.lanelet_id}"
            )

    start_position, start_heading = midpoint_pose(lanelets[0])
    goal_position, goal_heading = midpoint_pose(lanelets[-1])

    start_matches = network.find_lanelet_by_position([start_position])[0]
    goal_matches = network.find_lanelet_by_position([goal_position])[0]

    assert ROUTE[0] in start_matches
    assert ROUTE[-1] in goal_matches

    # Synthetic starting condition, not a measured CARLA actor.
    initial_state = InitialState(
        time_step=0,
        position=start_position,
        orientation=start_heading,
        velocity=INITIAL_SPEED,
        acceleration=0.0,
        yaw_rate=0.0,
        slip_angle=0.0,
    )

    # Estimate a generous deadline using the complete lanelet lengths.
    route_length = sum(
        float(np.linalg.norm(np.diff(l.center_vertices, axis=0), axis=1).sum())
        for l in lanelets
    )

    if scenario.dt <= 0:
        raise ValueError("Scenario timestep must be positive")

    time_limit_s = 3.0 * route_length / INITIAL_SPEED + 10.0
    max_step = max(100, math.ceil(time_limit_s / scenario.dt))

    goal_state = CustomState(
        position=Circle(radius=1.5, center=goal_position),
        time_step=Interval(0, int(max_step)),
    )

    goal = GoalRegion(
        state_list=[goal_state],
        lanelets_of_goal_position={0: [ROUTE[-1]]},
    )

    planning_problem = PlanningProblem(
        planning_problem_id=1,
        initial_state=initial_state,
        goal_region=goal,
    )

    planning_problem_set = PlanningProblemSet()
    planning_problem_set.add_planning_problem(planning_problem)

    writer = CommonRoadFileWriter(
        scenario=scenario,
        planning_problem_set=planning_problem_set,
        author="Honors CARLA-Rulebook Integration",
        affiliation="Research prototype",
        source="CARLA Town01 / CommonRoad Scenario Designer 0.8.5",
        tags={Tag.URBAN},
    )

    writer.write_to_file(
        str(OUTPUT),
        OverwriteExistingFile.ALWAYS,
    )

    # Verify that the new file can be opened again.
    loaded_scenario, loaded_problems = CommonRoadFileReader(
        str(OUTPUT)
    ).open()

    assert len(loaded_scenario.lanelet_network.lanelets) == 300
    assert len(loaded_problems.planning_problem_dict) == 1

    loaded_problem = list(loaded_problems.planning_problem_dict.values())[0]

    print("\n===== TOWN01 PLANNING PROBLEM =====")
    print("Route:", ROUTE)
    print("Start position:", start_position)
    print("Start heading (rad):", start_heading)
    print("Initial speed (m/s):", INITIAL_SPEED)
    print("Goal position:", goal_position)
    print("Goal heading (rad):", goal_heading)
    print("Route centerline length (m):", route_length)
    print("Goal deadline (steps):", max_step)
    print("Lanelets:", len(loaded_scenario.lanelet_network.lanelets))
    print("Planning problems:", len(loaded_problems.planning_problem_dict))
    print(
        "Reloaded goal-lanelet metadata:",
        loaded_problem.goal.lanelets_of_goal_position,
    )
    print("Output:", OUTPUT)
    print("\nSUCCESS: Planning problem XML written and reloaded.")


if __name__ == "__main__":
    main()
