# CommonRoad Rulebook Planner — Honors Project Research Log

> **Project status:** Standalone CommonRoad study completed through Lessons 6–9, including a sampled-state safety audit. **CARLA integration is deliberately paused.** This document records the working setup, experiments, findings, limitations, and next research questions as of November 2026.
>
> **Scope:** This is the Honors project's own README, intended for `D:\carla-rulebook\README.md`. **Do not replace** the upstream `commonroad-rb-planner/README.md` or `autonomous_driving/README.md`.

## 1. Research purpose

This project explores whether changing the **priority relationships between driving rules** in a sampling-based Rulebook-aware motion planner changes the autonomous vehicle's selected and executed trajectories. The eventual research setting includes Scenic scenario generation and the CARLA driving simulator, but the experiments described here execute **only in CommonRoad**.

The repository studied is [CommonRoad/commonroad-rb-planner](https://github.com/CommonRoad/commonroad-rb-planner), associated with:

> Patrick Halder and Matthias Althoff, **“Sampling-Based Motion Planning with Preordered Objectives,”** *IEEE Intelligent Vehicles Symposium*, 2025. See the upstream repository for the authors' recommended citation.

**What was demonstrated:** The research planner can load an example CommonRoad scenario, generate feasible candidate trajectories, compare them using Rulebook priorities, advance a standalone receding-horizon simulation, and produce different motions under two controlled Rulebook configurations.

**What was NOT demonstrated:** Execution of these Rulebook-selected trajectories in CARLA, continuous-time safety, legal lane changes, or statistically generalizable results across scenarios.

## 2. Environments and file locations — read this first

Two separate Python environments are intentional:

| Environment | Operating system / role | Python | Typical path |
| --- | --- | --- | --- |
| CARLA and Scenic | Windows; existing driving-simulator exercises | 3.11 | `D:\carla-rulebook\.venv` |
| CommonRoad Rulebook | **WSL 2 Ubuntu**; standalone research experiments | 3.9, Conda env `rpp_solver` | Ubuntu environment |

**The working research repository for Lessons 7–9 lives on the Windows D: drive and is accessed from Ubuntu using `/mnt/d`:**

| Windows | WSL Ubuntu |
| --- | --- |
| `D:\carla-rulebook` | `/mnt/d/carla-rulebook` |
| `D:\carla-rulebook\external\commonroad-rb-planner` | `/mnt/d/carla-rulebook/external/commonroad-rb-planner` |
| `D:\carla-rulebook\external\commonroad-rb-planner\autonomous_driving` | `/mnt/d/carla-rulebook/external/commonroad-rb-planner/autonomous_driving` |

**Do not mix repository copies.** Earlier setup used an additional checkout at `~/commonroad-rb-planner` inside Ubuntu. Editing that checkout has **no effect** on scripts run from `/mnt/d/carla-rulebook/external/commonroad-rb-planner`. Every command in the remainder of this document targets the working `/mnt/d` checkout.

Expected working layout (abbreviated):

```text
D:\carla-rulebook\
├── README.md                         # This project-level document
├── .venv\                            # Existing Windows CARLA/Scenic venv
├── results\
└── external\
    └── commonroad-rb-planner\
        ├── rulebook\
        │   └── rulebook.py
        └── autonomous_driving\
            ├── main_run_scenario.py  # Original research entry point
            ├── lesson7_run.py
            ├── lesson8_compare.py
            ├── lesson9_repeated_experiments.py
            ├── lesson9_safety_validation.py
            ├── scenarios\
            ├── configurations\rulebooks\
            ├── rpp_solver\
            │   ├── rpp_solver.py
            │   ├── run_rpp_solver_as_mpc.py
            │   └── common\utils.py
            └── output\
                ├── lesson8\
                └── lesson9\
```

The `lesson*.py` files are project-specific experiment scripts, **not** files supplied by the upstream Rulebook repository. Keep copies of them in `autonomous_driving/` before reproducing results.

## 3. What we learned about the planner

### 3.1 Planning layers

- **Global route/reference path:** A map-level route through the lanelet network.
- **Reactive motion planning:** Samples and filters dynamically feasible candidate trajectories over a short horizon.
- **Rulebook comparison:** Evaluates prioritized objectives and identifies **non-dominated** candidate trajectories. A priority order need not reduce to a single weighted-sum cost.
- **Selection:** The provided research implementation may randomly select from the non-dominated set.
- **Receding-horizon simulation:** Executes/advances one simulated state, obtains a new planning state, then plans again.

Do not confuse the standalone CommonRoad state-advance loop with a CARLA vehicle-controller/physics loop. No CARLA vehicle was involved in Lessons 7–9.

### 3.2 Source interfaces examined

| Source location relative to `autonomous_driving/` | Purpose |
| --- | --- |
| `main_run_scenario.py` | Upstream demonstration entry point |
| `rpp_solver/rpp_solver.py` | `RPPSolver`, `PlanningSolution`, trajectory generation and Rulebook selection |
| `rpp_solver/run_rpp_solver_as_mpc.py` | Repeated planning / state-advance loop; returns the executed state sequence and timing/cost data |
| `rpp_solver/common/data.py` | Builds scenario-derived input data and reference path |
| `rpp_solver/common/utils.py` | Scenario load, reference path, obstacle and geometry utilities |
| `configurations/rulebooks/rb_preorder_relations.txt` | Original preorder Rulebook configuration used as a base |
| `../rulebook/rulebook.py` | Rulebook comparison implementation, in a **sibling directory** |

The central planner interface examined is `RPPSolver.plan(x_0, data)`. It requires CommonRoad-style vehicle state and scenario data, not native CARLA actor objects. A returned selected trajectory can be accessed through the solution's selected sample, but a future CARLA implementation still requires a state/map/obstacle adapter and trajectory tracker.

### 3.3 Rules encountered in the experiment

The observed configurations included **seven** rules:

| Rule name | Interpretation |
| --- | --- |
| `R1_Collision_Avoidance` | Penalize predicted collisions |
| `R8_Road_Boundaries` | Penalize road-boundary contacts/violations |
| `R6_Simple_Speed_Limit` | Penalize excess predicted speed |
| `R4_Comfort` | Penalize undesirable motion/comfort cost |
| `R9_Safety_Distance` | Penalize inadequate predicted obstacle separation |
| `R5_Corridor` | Penalize deviation from the configured corridor |
| `R7_Progress` | Reward/penalize trajectory progress according to its rule definition |

Different costs have different meanings and scales. **Do not directly add or numerically compare unlike rule costs unless the formalism defines that operation.** A rule cost reported as `None` in the original efficient comparison means **not evaluated**, not zero.

## 4. Reproducing the software environment

### 4.1 Start WSL Ubuntu (Windows PowerShell only)

```powershell
wsl --list --verbose
wsl -d Ubuntu
```

If Ubuntu has not been installed, open administrator PowerShell and use `wsl --install -d Ubuntu` (restart if prompted). After launching Ubuntu, **all remaining terminal commands in this section are Linux Bash commands**.

### 4.2 Basic system tools and Conda (Ubuntu Bash)

```bash
sudo apt update
sudo apt install -y git curl build-essential pkg-config graphviz libgraphviz-dev
```

Install Miniconda for the machine's architecture. On a normal x86-64 Windows/WSL PC:

```bash
cd ~
curl -L --fail --show-error --progress-bar \
  --connect-timeout 15 -o miniconda.sh \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda init bash
```

Create/activate the independent planner environment:

```bash
conda create -n rpp_solver python=3.9 -y
conda activate rpp_solver
python --version
```

Python 3.9 is end-of-life; this legacy environment is for isolated research reproduction, not production or internet-facing services.

### 4.3 Use ONE source checkout

For a fresh working copy (skip cloning if the D: drive repository already exists):

```bash
mkdir -p /mnt/d/carla-rulebook/external
cd /mnt/d/carla-rulebook/external
git clone https://github.com/CommonRoad/commonroad-rb-planner.git
cd commonroad-rb-planner
```

To record precisely which upstream revision was tested:

```bash
git rev-parse HEAD
```

Later, also record the committed status and source changes (the RoutePlanner fix below will make the worktree dirty):

```bash
git status --short
git diff -- autonomous_driving/rpp_solver/common/utils.py
```

### 4.4 PyGraphviz: working workaround for the actual compiler failure

The upstream autonomous-driving requirements pinned `pygraphviz==1.7`. In this WSL environment, `pip install` for both PyGraphviz 1.7 and 1.10 failed while compiling `gvRenderData` because the generated wrapper expected `unsigned int *` but Graphviz's headers expected `size_t *`.

**Working approach used:** Install precompiled PyGraphviz through the `conda-forge` channel, **not** through pip:

```bash
conda activate rpp_solver
conda install -c conda-forge "pygraphviz=1.10" "python=3.9"
python -c "import pygraphviz; print(pygraphviz.__version__)"
```

Do not subsequently allow the old pinned requirement to replace the Conda package. Create a temporary requirements copy that omits its `pygraphviz==1.7` line:

```bash
cd /mnt/d/carla-rulebook/external/commonroad-rb-planner
sed -E '/^[[:space:]]*pygraphviz[[:space:]]*==[[:space:]]*1\.7[[:space:]]*$/d' \
  autonomous_driving/requirements.txt > /tmp/lesson7-requirements.txt
```

The older dependency constraints trialed in the successful installation workflow were:

```bash
cat > lesson7-constraints.txt <<'EOF_CONSTRAINTS'
numpy==1.22.0
commonroad-io==2023.3
commonroad-route-planner==2024.1.1
commonroad-reactive-planner==2024.1
commonroad-drivability-checker==2024.1
EOF_CONSTRAINTS
```

Install the repository's local Rulebook package and remaining autonomous-driving requirements:

```bash
python -m pip install "pip<26" "setuptools>=68.2.2" wheel
python -m pip install -r rulebook/requirements.txt
python -m pip install -e ./rulebook
python -m pip install \
  -c lesson7-constraints.txt \
  -r /tmp/lesson7-requirements.txt \
  "commonroad-route-planner==2024.1.1" \
  "commonroad-drivability-checker==2024.1"
```

**Reproducibility warning:** The constraints above document the compatibility choices used during troubleshooting, not a verified full environment lockfile. For a truly exact replay, preserve `conda list --explicit`, `pip freeze`, and the Git revision from your **working** environment; package builds and availability can change. Do not attempt to blindly upgrade these research dependencies.

Verify before running the planner:

```bash
python -m pip check
python -c "import pygraphviz; print('PyGraphviz OK')"
cd /mnt/d/carla-rulebook/external/commonroad-rb-planner/autonomous_driving
export PYTHONPATH="$(realpath ../rulebook)${PYTHONPATH:+:$PYTHONPATH}"
python -c "from rulebook import Rulebook; from rpp_solver.rpp_solver import RPPSolver; print('Planner imports OK')"
```

**Why `PYTHONPATH` matters:** `rulebook/rulebook.py` is in a sibling folder to `autonomous_driving/`. Importing `Rulebook` from *inside* `rulebook/` does not prove the same import works from `autonomous_driving/`. Scripts from Lessons 8–9 add the sibling source path themselves; Lesson 7 needs the explicit shell setting.

### 4.5 Required RoutePlanner API compatibility fix

The upstream `autonomous_driving/rpp_solver/common/utils.py` had:

```python
route_planner = RoutePlanner(scenario.lanelet_network, planning_problem)
```

The installed RoutePlanner version expects the full `Scenario`, so in **the exact repository copy you are running**, replace that line with:

```python
route_planner = RoutePlanner(scenario, planning_problem)
```

Safely patch once, with a backup:

```bash
cd /mnt/d/carla-rulebook/external/commonroad-rb-planner/autonomous_driving
python - <<'PY'
from pathlib import Path
p = Path('rpp_solver/common/utils.py')
old = 'RoutePlanner(scenario.lanelet_network, planning_problem)'
new = 'RoutePlanner(scenario, planning_problem)'
s = p.read_text(encoding='utf-8')
if old in s:
    assert s.count(old) == 1
    p.with_suffix('.py.bak').write_text(s, encoding='utf-8')
    p.write_text(s.replace(old, new), encoding='utf-8')
elif new not in s:
    raise RuntimeError('Unexpected source: inspect create_reference_path manually')
print('Current RoutePlanner call:', new)
PY
```

Confirm:

```bash
grep -n 'route_planner = RoutePlanner' rpp_solver/common/utils.py
python -m py_compile rpp_solver/common/utils.py
```

The standalone scenario reference path test returned **1,696 path points** after this correction. The message `Lanelets_of_goal_position not given` was non-fatal in that run.

### 4.6 Archive the exact working environment

Run within the activated Conda environment:

```bash
cd /mnt/d/carla-rulebook/external/commonroad-rb-planner
conda list --explicit > commonroad-conda-explicit.txt
python -m pip freeze > commonroad-pip-freeze.txt
python -m pip check
```

These files preserve much more specific dependency information than this narrative README. Keep them with the research artifacts for future reproduction; they may contain platform-specific build URLs.

## 5. Re-run the experiments (all commands in WSL Ubuntu)

Activate and choose the **same** working checkout once per shell session:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rpp_solver
cd /mnt/d/carla-rulebook/external/commonroad-rb-planner/autonomous_driving
export PYTHONPATH="$(realpath ../rulebook)${PYTHONPATH:+:$PYTHONPATH}"
```

### Lesson 7 — first successful standalone Rulebook run

Script: `autonomous_driving/lesson7_run.py` (the project-specific file created during Lesson 7).

```bash
python -u lesson7_run.py
```

- Example scenario: `ZAM_Paper-1_1_T-1.xml`.
- Rulebook: `configurations/rulebooks/rb_preorder_relations.txt`.
- Comparison methods: `trajectory_comparison_method=2`, `rule_comparison_method=2`.
- Initial experiment: 10 planning iterations, 11 recorded vehicle states, 0.1-second nominal timestep.
- Log showed around 305–306 feasible candidates per iteration and 2–3 non-dominated candidates in the original preorder setup.
- The program's reported selection time must **not** be confused with total planning time: total per-iteration processing was about 1.0–1.1 seconds in the logged run.

The executed trajectory is a **CommonRoad state-advance sequence**, not a CARLA physics rollout.

### Lesson 8 — two controlled Rulebook variants

Script: `autonomous_driving/lesson8_compare.py`.

```bash
python -u lesson8_compare.py
```

This script derives two **seven-rule** configurations from the upstream preorder file, retaining the same rule set while reversing the priority relation between:

- `R9_Safety_Distance` and `R5_Corridor`.

**B1 `safety_priority`:** R9 takes priority over R5.

**B2 `corridor_priority`:** R5 takes priority over R9.

Outputs: `autonomous_driving/output/lesson8/`:

```text
safety_priority_rulebook.txt
corridor_priority_rulebook.txt
safety_priority_trajectory.csv
corridor_priority_trajectory.csv
safety_priority_summary.json
corridor_priority_summary.json
comparison_summary.csv
trajectory_comparison.png
speed_comparison.png
```

The original `rb_preorder_relations.txt` is **not overwritten**.

### Lesson 9A — repeated matched seeds (10 steps)

Script: `autonomous_driving/lesson9_repeated_experiments.py`.

```bash
python -u lesson9_repeated_experiments.py --steps 10 --seeds 0 1 2 3 4
```

Precondition: the two Lesson 8 Rulebook files exist in `output/lesson8/`. The script copies them into the Lesson 9 output tree and records hashes, making the priority configuration traceable.

Outputs include:

```text
output/lesson9/steps_10/
├── run_metadata.json
├── paired_results.csv
├── aggregate_summary.json
├── failed_seeds.json
├── final_separation_by_seed.png
├── rulebooks/
└── seed_000/                           # and seeds 001–004
    ├── safety_priority_trajectory.csv
    ├── corridor_priority_trajectory.csv
    ├── safety_priority_planning_audit.csv
    ├── corridor_priority_planning_audit.csv
    ├── safety_priority_audit_errors.json
    ├── corridor_priority_audit_errors.json
    ├── aligned_separation.csv
    └── ...
```

The Lesson 9 subclass calls the unmodified selection algorithm, **then** evaluates all configured rules on the already selected candidate. The `lazy_*` columns record scores computed during selection; `full_*` columns are the additional post-selection evaluation. They are **costs on future planning candidates**, not realized cumulative traffic violations. Check each `*_audit_errors.json` before interpreting full-score coverage.

### Lesson 9B — longer paired runs (20 steps)

```bash
python -u lesson9_repeated_experiments.py --steps 20 --seeds 0 1 2
```

Outputs in `output/lesson9/steps_20/`, leaving 10-step outputs unchanged. The command generates trajectory/speed/road-context plots for the first seed in the run and the aggregate plot across seeds.

### Lesson 9C — sampled-state safety audit of the 20-step runs

Script: `autonomous_driving/lesson9_safety_validation.py`.

```bash
python -u lesson9_safety_validation.py --steps 20 --seeds 0 1 2
```

**Precondition:** Complete the 20-step experiment above; this script **reads the already recorded trajectories**, it does not rerun the planner.

Results:

```text
output/lesson9/steps_20/safety_validation/
├── per_step_validation.csv
├── run_summary.csv
├── validation_metadata.json
├── validation_warnings.json
├── seed_000_footprints.png
├── seed_001_footprints.png
└── seed_002_footprints.png
```

It reconstructs approximate ego and obstacle polygons, checks sampled road-boundary contact against CommonRoad's boundary collision object, computes polygon-to-polygon clearance to scenario obstacles, and flags missing/extrapolated obstacle states. The audit used **1.0 m** as the sampled-state safety-distance threshold, recorded at **0.1-second** intervals. Check `validation_warnings.json` and coverage fields before drawing conclusions.

## 6. Experiment results (measured, not illustrative)

### 6.1 Lesson 7: standalone planner baseline

The original ten-iteration preorder run recorded 11 states. It generated about 305–306 feasible candidates per iteration, reducing them to 2–3 non-dominated candidates. Initial speed was **13.06 m/s** and the final reported speed was **12.42 m/s**. Full planning time was about 1.0–1.1 seconds per planning iteration, while reported Rulebook comparison/selection was substantially shorter. This does not establish real-time performance at 0.1-second replanning intervals.

### 6.2 Lesson 8: one seed, 10 steps

| Measure | Safety priority | Corridor priority |
| --- | ---: | ---: |
| Final X (m) | 56.5061 | 56.3936 |
| Final Y (m) | 0.8356 | -0.0173 |
| Final speed (m/s) | 12.8141 | 12.3230 |
| Distance traveled (m) | 12.9647 | 12.8173 |
| Final selected future R9 cost | 0.000 | 0.7625 |
| Final selected future R5 cost | 3.4063 | 0.000 |

**Final position separation:** 0.8603 m after 1.0 simulated second. These values illustrate a measured tradeoff in *selected future-trajectory costs*. They do not imply either configuration is globally safer.

### 6.3 Lesson 9: repeated seeds, 10 steps

Five of five paired seeds succeeded. Reported aggregate results:

- Mean final separation: **0.8558657 m**.
- Median: **0.8586467 m**.
- Range: **0.8472052–0.8602745 m**.
- Mean **discrete** Hausdorff distance: **0.8558657 m**.

These are descriptive results for a single scenario and a small seed set; not a statistical significance result.

### 6.4 Lesson 9: repeated seeds, 20 steps

Three of three paired seeds succeeded. Reported aggregate results:

| Metric | Result |
| --- | ---: |
| Mean final separation | **3.2441991 m** |
| Median final separation | 3.2445603 m |
| Minimum final separation | 3.1890260 m |
| Maximum final separation | 3.2990110 m |
| Mean discrete Hausdorff | 3.2441991 m |

The increased separation is observed over a longer simulated horizon (2.0 seconds), **not** evidence of a proportional change in the algorithm's sensitivity. In the road-context plots, the safety-priority path moves toward positive Y while the corridor-priority path remains near the initial lateral band. A trajectory crossing an internal lanelet boundary is not by itself a collision with the drivable-road edge or proof of a legal/illegal lane change.

### 6.5 Final sampled-state safety validation of the 20-step runs

For each rulebook and each of seeds 0, 1 and 2, 21 states were checked: **126 ego-state records** in total. The script evaluated **two available obstacles for each record (252 obstacle-footprint checks)**. There were no missing or extrapolated obstacle states and `validation_warnings.json` was empty.

| Seed | Rulebook | Minimum sampled obstacle clearance (m) | Sampled obstacle-contact states | Sampled road-boundary-contact states | States below 1 m |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | Safety | 5.1836 | 0 | 0 | 0 |
| 0 | Corridor | 6.2495 | 0 | 0 | 0 |
| 1 | Safety | 5.3041 | 0 | 0 | 0 |
| 1 | Corridor | 6.2495 | 0 | 0 | 0 |
| 2 | Safety | 5.4310 | 0 | 0 | 0 |
| 2 | Corridor | 6.2495 | 0 | 0 | 0 |

The minimum clearance in each run occurred at the **final sampled state (t = 2.0 s)**. Interestingly, the safety-priority paths had **lower sampled obstacle clearance** than their corridor-priority counterparts even though **none** crossed the 1 m threshold. This is compatible with the fact that a Rulebook priority influences *planning comparisons* rather than guaranteeing a single rule's best realized score.

**Important distinction:** The predicted future R9 cost at the final planning step can be positive even when the executed 0–2 s states all have more than 1 m clearance. The two evaluations have different time horizons and possibly different obstacle/context sets. Do not conflate them.

## 7. Evaluation metrics and interpretation

- **Final position separation:** Euclidean distance between the two final X–Y positions.
- **Maximum time-aligned separation:** Largest Euclidean distance between positions recorded at matching timesteps.
- **Discrete symmetric Hausdorff distance:** Largest nearest-neighbor distance between two *sampled sets* of trajectory positions, ignoring timestamp alignment; not a continuous-time metric.
- **Final speed difference:** Safety-priority speed minus corridor-priority speed (positive means safety-priority faster).
- **Future rule costs:** Rule-specific evaluation of a selected planning-horizon candidate; not cumulative realized-path violations.
- **Sampled collision/contact flags:** Geometry contacts at recorded times only; not a swept continuous-time collision proof.

Use `run_metadata.json`, `paired_results.csv`, the per-step planning audit files, and `validation_metadata.json` when reporting experimental details. In particular, confirm the Rulebook hashes and that no full-rule-evaluation errors or missing obstacle states were reported.

## 8. Known limitations / unresolved research questions

1. **One scenario only:** The findings concern `ZAM_Paper-1_1_T-1`, not all roads or obstacle interactions.
2. **Short experiments:** 10/20 steps correspond to 1/2 simulated seconds at 0.1 s per step.
3. **Small seed counts:** Five matched seeds at 10 steps and three at 20 steps are useful exploratory checks but not an inferential study.
4. **Selection randomness:** Matching seeds controls the initial generator but does not ensure the candidate sets remain identical once trajectories diverge.
5. **Safety sampling:** With a 0.1 s audit interval, collisions between samples may be missed.
6. **Geometry versus traffic law:** No test here establishes that lane changes are legally permitted, nor compliance with every road rule.
7. **Obstacle forecasting:** The audit checks the scenario's available obstacle states at each sampled timestep. The planner's prediction and filtering may not match the audit's full set of obstacles.
8. **Timing:** The total planning iteration was ~1 s in the initial experiment, not yet suitable for strict 0.1 s real-time execution as observed.
9. **Cross-platform interfaces:** CommonRoad expects CommonRoad states, lanelet geometry, obstacles, and a reference route; CARLA uses different coordinates, maps and vehicle controls. These interfaces have **not** been connected or validated.

### Open questions to investigate later

- Why does the safety-priority path diverge laterally while its sampled obstacle clearance is sometimes lower than the corridor-priority path?
- At what later horizon or in which traffic scenarios does the desired rule priority improve realized measured safety margins?
- Are the new lateral maneuvers legal under lanelet connectivity/marking rules?
- Which Rulebook comparisons drive changes at the first divergence point, and are all viable candidates safe under hard constraints?
- How can the computational planning cycle be integrated at a practical replanning frequency?

## 9. Troubleshooting record

| Symptom | Root cause found / working response |
| --- | --- |
| `conda: command not found` in **Windows PowerShell** | The research Conda environment is installed **inside WSL Ubuntu**, not Windows. Launch Ubuntu. |
| Miniconda `curl -fsSLo` looks frozen | Silent download hides progress. Use `curl --progress-bar` with timeouts. |
| `from rulebook import Rulebook` fails from repository root | The source lives in `rulebook/rulebook.py` and import lookup depends on current path. From `autonomous_driving`, add the sibling `rulebook/` directory to `PYTHONPATH`. |
| PyGraphviz 1.7 or 1.10 pip wheel build fails at `gvRenderData` | Older C wrapper incompatible with installed Graphviz headers. The **conda-forge precompiled PyGraphviz package** resolved this in the existing environment. |
| `AttributeError: 'LaneletNetwork' object has no attribute 'lanelet_network'` | Patch the research helper to call `RoutePlanner(scenario, planning_problem)` instead of passing `scenario.lanelet_network`. |
| Patching `~/commonroad-rb-planner` has no effect on `/mnt/d/...` run | Two different checkouts. Patch the **same exact copy** from which Python executes the script. |
| Empty Lesson 8 result directory and empty log | Earlier incomplete experiment script did not reach `main()`. Use the corrected `lesson8_compare.py` with an explicit `if __name__ == '__main__': main()` and visible progress messages. |
| Lazy rule costs are `None` | Efficient comparison skipped that rule's evaluation. Use Lesson 9's **post-selection full** audit and inspect audit-error files. |

Never treat `pip check` alone as proof that every required module is present: it reports only inconsistencies among already installed distributions. Verify imports and an actual scenario execution too.

## 10. Pause point and next phase (not yet started)

**The CommonRoad-only validation is complete. Pause here.** Do not start coding a CARLA integration yet.

Before choosing an integration architecture, review the Honors project's advisor-recommended **Planning Algorithms paper and accompanying code** (exact paper/repository references still to be supplied). Compare its assumptions, data interfaces, trajectory generation, constraints, and control architecture with the existing CommonRoad Rulebook implementation. The student also plans to share additional CARLA-specific ideas before committing to an integration design.

Any later integration plan should explicitly address:

- CARLA / Scenic scenario lifecycle and ownership of vehicle control;
- conversion between CARLA's map/vehicle/obstacle states and CommonRoad objects;
- time synchronization and coordinate/heading conventions;
- converting chosen CommonRoad trajectories into CARLA control inputs;
- validation of executed trajectories and robust handling of planner infeasibility;
- fair, reproducible comparisons of Rulebooks **within CARLA**.

## 11. References

- [CommonRoad Rulebook Planner — official repository](https://github.com/CommonRoad/commonroad-rb-planner)
- [PyGraphviz 1.10 installation documentation — Conda Forge guidance](https://pygraphviz.github.io/documentation/pygraphviz-1.10/install.html)
- [CommonRoad Route Planner — repository](https://github.com/CommonRoad/commonroad-route-planner)
- Local evidence: experiment output CSV/JSON and PNG files under `autonomous_driving/output/lesson8/` and `autonomous_driving/output/lesson9/`.

---

**Reproduction quick reference (inside WSL Ubuntu):**

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rpp_solver
cd /mnt/d/carla-rulebook/external/commonroad-rb-planner/autonomous_driving
export PYTHONPATH="$(realpath ../rulebook)${PYTHONPATH:+:$PYTHONPATH}"
python -u lesson7_run.py
python -u lesson8_compare.py
python -u lesson9_repeated_experiments.py --steps 10 --seeds 0 1 2 3 4
python -u lesson9_repeated_experiments.py --steps 20 --seeds 0 1 2
python -u lesson9_safety_validation.py --steps 20 --seeds 0 1 2
```

**Note:** These commands overwrite outputs in their named experiment directories when rerun. Copy/rename any results you wish to archive beforehand.
