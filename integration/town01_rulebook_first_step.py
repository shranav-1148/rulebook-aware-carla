"""Milestone 3C: one Rulebook-aware motion-planning call on synthetic Town01.

Project-specific, AI-assisted integration code (not upstream CommonRoad code).
Input:  integration/maps/Town01.route_38_40_58_60.xml
Output: integration/output/town01_rulebook_first_step/

The CARLA server and Scenic are NOT used here. Nothing controls a vehicle.
The research planner source tree and its configuration files are NOT modified.
See town01_rulebook_first_step_PROVENANCE.md for source/API attribution.
"""

import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent
RESEARCH_ROOT = PROJECT_ROOT / "external" / "commonroad-rb-planner" / "autonomous_driving"
RULEBOOK_ROOT = RESEARCH_ROOT.parent / "rulebook"
SCENARIO_FILE = BASE / "maps" / "Town01.route_38_40_58_60.xml"
RULEBOOK_FILE = RESEARCH_ROOT / "configurations" / "rulebooks" / "rb_preorder_relations.txt"
OUT = BASE / "output" / "town01_rulebook_first_step"
SEED = 0
EXPECTED_LANELETS = 300


def sha256_file(path):
    """Hash inputs so subsequent experiments can identify what was tested."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serialize_costs(costs):
    """Preserve None: in the upstream lazy evaluator it means NOT evaluated."""
    result = {}
    for name, value in costs.items():
        if value is None:
            result[str(name)] = None
        else:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"Non-finite cost for {name}: {number}")
            result[str(name)] = number
    return result


def state_records(states, dt):
    """Extract numeric states from the selected *predicted* trajectory."""
    records = []
    for state in states:
        pos = state.position
        row = {
            "time_step": int(state.time_step),
            "time_s": float(state.time_step) * float(dt),
            "x_m": float(pos[0]),
            "y_m": float(pos[1]),
            "orientation_rad": float(state.orientation),
            "velocity_mps": float(state.velocity),
            "acceleration_mps2": (
                None if getattr(state, "acceleration", None) is None
                else float(state.acceleration)
            ),
        }
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Non-finite trajectory value {key}={value}")
        records.append(row)
    if len(records) < 2:
        raise RuntimeError("Selected trajectory has fewer than two states")
    return records


def make_empty_safe_r9_class(upstream_r9_class):
    """Define a project-local R9 subclass; leave upstream Python files unchanged.

    Only the no-prediction case is new (cost 0). For nonempty predictions,
    evaluation delegates to the original research implementation.
    """
    class R9WithEmptyPredictionGuard(upstream_r9_class):
        def evaluate(self, sample):
            if self.data is None:
                raise RuntimeError("R9 has not received planning data")
            if len(self.data.predictions) == 0:
                return 0.0
            return super().evaluate(sample)

    return R9WithEmptyPredictionGuard


def validate_files():
    for description, path in (
        ("Town01 planning-problem XML", SCENARIO_FILE),
        ("Rulebook configuration", RULEBOOK_FILE),
        ("research planner entry module", RESEARCH_ROOT / "rpp_solver" / "rpp_solver.py"),
        ("research configuration defaults", RESEARCH_ROOT / "configurations" / "defaults" / "planning.yaml"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}")
    # The copy used during Lessons 7-9 already had an API compatibility fix.
    # Do not silently patch upstream files from this independent experiment.
    utils_path = RESEARCH_ROOT / "rpp_solver" / "common" / "utils.py"
    if not utils_path.is_file():
        raise FileNotFoundError(f"Missing research utility module: {utils_path}")
    if "RoutePlanner(scenario.lanelet_network, planning_problem)" in utils_path.read_text(encoding="utf-8"):
        raise RuntimeError(
            "The chosen research checkout still uses the incompatible old "
            "RoutePlanner(scenario.lanelet_network, planning_problem) call. "
            "Use the same tested checkout/setup as Lesson 7; this script will "
            "not modify any upstream source files."
        )


def save_plot(reference_path, records, output_file):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = np.asarray(reference_path, dtype=float)
    actual = np.array([[r["x_m"], r["y_m"]] for r in records], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ref[:, 0], ref[:, 1], "--", linewidth=1.5, label="Route reference path")
    ax.plot(actual[:, 0], actual[:, 1], marker=".", markersize=4,
            linewidth=2, label="Selected predicted trajectory")
    ax.scatter(actual[0, 0], actual[0, 1], s=80, marker="o", label="Initial state", zorder=5)
    ax.scatter(actual[-1, 0], actual[-1, 1], s=80, marker="x", label="End of prediction", zorder=5)
    pad = 12.0
    ax.set_xlim(float(actual[:, 0].min()) - pad, float(actual[:, 0].max()) + pad)
    ax.set_ylim(float(actual[:, 1].min()) - pad, float(actual[:, 1].max()) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("CommonRoad X (m)")
    ax.set_ylabel("CommonRoad Y (m)")
    ax.set_title("Town01: Rulebook-selected prediction (NOT CARLA execution)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_file, dpi=160)
    plt.close(fig)


def main():
    validate_files()

    # The original planner loads its reactive-planner YAML using relative paths.
    # Changing our process's CWD is NOT editing the repository.
    os.chdir(RESEARCH_ROOT)
    sys.path.insert(0, str(RESEARCH_ROOT))
    sys.path.insert(0, str(RULEBOOK_ROOT))

    import numpy as np
    from commonroad.common.file_reader import CommonRoadFileReader
    from rpp_solver.common.configuration_builder import ConfigurationBuilder
    from rpp_solver.common.data import InputDataHandler
    from rpp_solver.rpp_solver import RPPSolver
    from rpp_solver.rules.R9_safety_distance import R9_Safety_Distance

    # Seed after importing RPPSolver: upstream seeds `random` at import time.
    random.seed(SEED)
    np.random.seed(SEED)

    scenario, problem_set = CommonRoadFileReader(str(SCENARIO_FILE)).open()
    problems = list(problem_set.planning_problem_dict.values())
    if len(problems) != 1:
        raise RuntimeError(f"Expected exactly one Town01 planning problem; got {len(problems)}")
    problem = problems[0]
    if len(scenario.lanelet_network.lanelets) != EXPECTED_LANELETS:
        raise RuntimeError("Lanelet count changed; inspect the scenario before continuing")
    if len(scenario.obstacles) != 0:
        raise RuntimeError("This first-step experiment must be obstacle-free")

    # Build a configuration from the research project's existing DEFAULTS,
    # without loading the unrelated ZAM_Paper scenario or changing its YAML.
    config = ConfigurationBuilder.build_configuration(
        SCENARIO_FILE.stem, path_root=str(RESEARCH_ROOT)
    )
    config.rulebook.trajectory_comparison_method = 2
    config.rulebook.rule_comparison_method = 2
    config.planning.time_steps_computation = 1
    config.planning.replanning_frequency = 1
    config.planning.extrapolate_obstacle_prediction = True
    config.debug.safe_reference_trajectory = False
    config.debug.plot_rulebook = False
    config.debug.plot_solution = False
    config.debug.plot_scenario = False
    config.debug.create_video = False

    if not math.isclose(float(scenario.dt), float(config.planning.dt),
                        abs_tol=1e-9, rel_tol=0.0):
        raise RuntimeError(
            f"Timestep mismatch: scenario.dt={scenario.dt}, "
            f"planner config.dt={config.planning.dt}"
        )

    # This is precisely the initialization sequence used by the original
    # run_rpp_solver_as_mpc (see its lines 19-22), but we call plan() ONCE.
    print("[3C] Building the planner's own reference path and input data...", flush=True)
    handler = InputDataHandler(scenario, problem, config)
    planner = RPPSolver(str(RULEBOOK_FILE), scenario, config)

    # Local experimental compatibility guard for the obstacle-free case.
    # R9 upstream calls max(costs) without a default; when there are no
    # predictions this raises ValueError. Zero is the neutral violation cost.
    # All nonempty-prediction evaluations use the untouched upstream method.
    rule_name = "R9_Safety_Distance"
    original_rule = planner.rulebook.used_rules.get(rule_name)
    if original_rule is None or not isinstance(original_rule, R9_Safety_Distance):
        raise RuntimeError("Expected original R9_Safety_Distance in Rulebook")
    GuardedR9 = make_empty_safe_r9_class(R9_Safety_Distance)
    planner.rulebook.used_rules[rule_name] = GuardedR9(config)
    print("[3C] Applied local R9 guard: no predictions => cost 0; "
          "otherwise upstream evaluation unchanged.", flush=True)

    planner.initialize_reactive_planner(handler)
    x0 = planner.reactive_planner.x_0
    current_step = int(x0.time_step)
    data = handler.get(current_step, x0)
    print(f"[3C] Predicted obstacle count: {len(data.predictions)}", flush=True)
    if data.predictions:
        raise RuntimeError("This checkpoint intentionally requires no obstacle predictions")

    print("[3C] Running exactly one Rulebook-aware planning call...", flush=True)
    started = time.perf_counter()
    solution = planner.plan(x0, data)
    planning_wall_s = time.perf_counter() - started

    selected = solution.selected_sample
    costs = serialize_costs(selected.costs)
    records = state_records(selected.traj_cart.state_list, config.planning.dt)
    lanelet_ids = [int(n) for n in handler.route.lanelet_ids]
    if not lanelet_ids:
        raise RuntimeError("The research planner's route contains no lanelet IDs")

    goal_times = [getattr(goal_state, "time_step", None)
                  for goal_state in problem.goal.state_list]
    goal_start = [getattr(t, "start", None) for t in goal_times if t is not None]
    if any(start == 0 for start in goal_start):
        print("WARNING: the synthetic goal window begins at step 0. "
              "Upstream R3_Mission uses the goal window START to compute "
              "local progress targets; interpret R3 scores with caution.", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "selected_predicted_trajectory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "experiment": "Town01 one-step synthetic Rulebook-aware planning (no CARLA control)",
        "input_scenario": str(SCENARIO_FILE),
        "input_scenario_sha256": sha256_file(SCENARIO_FILE),
        "rulebook_file": str(RULEBOOK_FILE),
        "rulebook_sha256": sha256_file(RULEBOOK_FILE),
        "seed": SEED,
        "local_rule_compatibility": {
            "rule": "R9_Safety_Distance",
            "empty_prediction_behavior": "return 0.0",
            "nonempty_prediction_behavior": "delegate to original upstream R9.evaluate",
            "research_repository_modified": False,
        },
        "scenario_dt_s": float(scenario.dt),
        "planning_horizon_steps": int(config.planning.planning_horizon),
        "initial_position_m": [float(z) for z in x0.position[:2]],
        "initial_speed_mps": float(x0.velocity),
        "route_lanelet_ids_from_upstream_handler": lanelet_ids,
        "reference_path_points_from_upstream_handler": int(len(handler.route.reference_path)),
        "selected_candidate_id": int(selected.id),
        "feasible_candidates": len(solution.samples),
        "non_dominated_candidates": len(solution.dominating_samples),
        "planning_wall_time_s": planning_wall_s,
        "rule_evaluation_time_s": float(solution.rule_evaluation_time),
        "selection_time_reported_s": float(solution.find_dominants_time_total),
        "selected_lazy_rule_costs": costs,
        "selected_prediction_states": len(records),
        "first_predicted_state": records[0],
        "next_predicted_state": records[1],
        "last_predicted_state": records[-1],
        "goal_time_window_starts": goal_start,
        "important_limitations": [
            "Predicted trajectory only; no vehicle was driven in CARLA",
            "No dynamic or static obstacles in this scenario",
            "Saved Town01 XML uses lanelet-derived polygon goal, not original circle",
            "Town01.rpp_compat.xml maps curb line-marking semantics to unknown",
            "None in lazy rule costs means not evaluated, not zero",
            "The input goal window may begin at step 0, biasing R3_Mission cost",
            "The upstream handler constructs its own reference path; this is not necessarily the Milestone 3B CSV",
            "Project-local R9 guard returns zero for an empty obstacle-prediction list; this differs from the original upstream error behavior",
        ],
    }
    summary_path = OUT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot_path = OUT / "selected_predicted_trajectory.png"
    save_plot(handler.route.reference_path, records, plot_path)

    print("\n===== TOWN01 FIRST RULEBOOK PLANNING STEP =====")
    print("Lanelets:", lanelet_ids)
    print("Feasible samples:", len(solution.samples))
    print("Non-dominated samples:", len(solution.dominating_samples))
    print("Selected candidate ID:", selected.id)
    print(f"Wall planning time: {planning_wall_s:.3f} s")
    print("Lazy rule costs (None = not evaluated):", costs)
    print("Selected predicted states:", len(records))
    print("First predicted position:", [records[0]["x_m"], records[0]["y_m"]])
    print("Next predicted position:", [records[1]["x_m"], records[1]["y_m"]])
    print("End predicted position:", [records[-1]["x_m"], records[-1]["y_m"]])
    print("CSV:", csv_path)
    print("Summary:", summary_path)
    print("Plot:", plot_path)
    print("SUCCESS: One Rulebook planning call completed (no CARLA execution).")


if __name__ == "__main__":
    main()
