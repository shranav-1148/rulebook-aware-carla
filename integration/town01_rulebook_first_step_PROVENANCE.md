# Provenance: Milestone 3C, one Rulebook planning step on Town01

## Files and authorship

- `town01_rulebook_first_step.py`: AI-assisted, project-specific adapter. Written as independent glue code for `D:/carla-rulebook/integration`; **not** a copy of the research code. It reads a previously generated CommonRoad XML, uses the existing planner through its public Python classes, runs one planning step, and writes outputs only under `integration/output/town01_rulebook_first_step/`.
- `test_town01_rulebook_first_step.py`: AI-assisted, project-specific unit tests for result serialization. It does not invoke the research planner.
- Neither of these scripts modifies the advisor's `tichakornw/planning`, the CommonRoad Rulebook Planner checkout, or any installed packages. `os.chdir` changes only the current Python process's working directory, needed by upstream config loading.

## Source dependencies, exact APIs, and attribution

1. Patrick Halder and Matthias Althoff, **Sampling-Based Motion Planning with Preordered Objectives**, IV 2025; original research implementation, `CommonRoad/commonroad-rb-planner` — https://github.com/CommonRoad/commonroad-rb-planner
   - `autonomous_driving/rpp_solver/rpp_solver.py`: `RPPSolver(rulebook_path, scenario, config)`, `initialize_reactive_planner(data_handler)`, `plan(x_0, data)`, and the `PlanningSolution` fields used for results: https://github.com/CommonRoad/commonroad-rb-planner/blob/main/autonomous_driving/rpp_solver/rpp_solver.py
   - `autonomous_driving/rpp_solver/common/data.py`: `InputDataHandler(scenario, planning_problem, config)` and `.get(step, x_0)`; generates its **own** route and reference path: https://github.com/CommonRoad/commonroad-rb-planner/blob/main/autonomous_driving/rpp_solver/common/data.py
   - `autonomous_driving/rpp_solver/common/configuration_builder.py`: `ConfigurationBuilder.build_configuration(name, path_root=...)`, which loads research configuration defaults: https://github.com/CommonRoad/commonroad-rb-planner/blob/main/autonomous_driving/rpp_solver/common/configuration_builder.py
   - `autonomous_driving/rpp_solver/run_rpp_solver_as_mpc.py`: original initialization sequence on lines 19–22 and planning on lines 45–50. This adapter calls `planner.plan(...)` once directly rather than copying the MPC loop: https://github.com/CommonRoad/commonroad-rb-planner/blob/main/autonomous_driving/rpp_solver/run_rpp_solver_as_mpc.py
   - `autonomous_driving/rpp_solver/common/mission.py`: `R3_Mission` uses the **start** of the goal time window; an interval starting at step zero is a limitation in the synthetic Town01 file: https://github.com/CommonRoad/commonroad-rb-planner/blob/main/autonomous_driving/rpp_solver/common/mission.py
2. CommonRoad-IO scenario reader: https://github.com/CommonRoad/commonroad-io ; `CommonRoadFileReader` loads Town01 and the saved `PlanningProblem`.
3. NumPy, Matplotlib, and the Python standard library for output formatting and plotting.

All third-party algorithms, Rulebook definitions, rule evaluation, feasibility checks, motion sampling, route generation, and selection belong to the respective upstream dependencies, not to this adapter. Preserve their license notices and citation requirements when distributing research results; use the local dependency checkout's license texts as authority.

## Project-specific inputs and preconditions

- `integration/maps/Town01.route_38_40_58_60.xml`: created in previous project step with a synthetic ego at (366.411, 2.003), speed 8 m/s, polygonal goal lanelet 60; 300 lanelets from CARLA Town01 OpenDRIVE conversion.
- Original `Town01.commonroad.xml` was created using CommonRoad Scenario Designer 0.8.5; the project-specific `Town01.rpp_compat.xml` replaced unsupported `curb` lane-marking labels with `unknown` in a copy for older CommonRoad-IO compatibility. This **can change semantic rule interpretation** and is not a validated lane-boundary-conformance model.
- Research environment: WSL Conda `rpp_solver`, Python 3.9. The research checkout previously needed the older RoutePlanner compatibility change. This script **refuses** to patch the upstream checkout itself. If its API is still incompatible, stop and inspect the existing Lesson 7 setup.
- Default Rulebook: `configurations/rulebooks/rb_preorder_relations.txt`, unmodified. The adapter sets efficient comparison methods to 2 in the in-memory configuration; the underlying research YAML is not altered.
- Scenario XML is read unchanged. The adapter intentionally does **not** correct the in-memory goal deadline because this first experiment tests the exact previously serialized problem. Its `Interval(0, 999)` makes the upstream `R3_Mission` target more aggressive than likely intended; do **not** interpret R3 costs as a calibrated mission-performance result.
- The upstream data handler may replan its own reference path and resample it differently from the earlier `reference_path.csv`; this script records its actual lanelet IDs and reference path point count.

## Validation performed in the artifact generation runtime

- `python -m py_compile town01_rulebook_first_step.py` succeeded.
- `python -m unittest discover -s /mnt/data -p test_town01_rulebook_first_step.py -v` passed (4 tests, pure-Python helper behavior only).
- The full planner, the user's XML, CARLA, and Scenic **were not available** for live execution here. No runtime success, candidate counts, rule costs, collision-free execution, or physical feasibility claim is made before user-side testing.

## Results to preserve from the first user-side run

- Input scenario and Rulebook SHA-256 hashes from `summary.json`.
- `selected_predicted_trajectory.csv`: selected **predicted** trajectory, not executed states.
- `selected_predicted_trajectory.png`: local view of the selected prediction versus the upstream handler's reference path.
- `summary.json`: candidate counts, selected candidate, raw lazy Rulebook costs (null means *not evaluated*), timing, seed and explicit limitations.
- For exact reproduction, capture `git -C external/commonroad-rb-planner rev-parse HEAD`, `git status --short`, and environment package versions from the actual workstation.

## File SHA-256 (generation)

- `town01_rulebook_first_step.py`: `1507c357460140ff3ef3d72410d8bf4ed7fc32df04d7f6a3f0191720b70bd985`
- `test_town01_rulebook_first_step.py`: `1b061c7e44908d24e2ac72f49397e8619d2c39ace3f548ce9153b7ae11602a08`
