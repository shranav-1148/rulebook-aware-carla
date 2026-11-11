# Milestone 2A — CARLA OpenDRIVE road map to CommonRoad lanelets

**Status:** AI-assisted research prototype for review. Map-conversion implementation is provided by **CommonRoad Scenario Designer**; these two files call documented public APIs. **No** third-party research source file was copied into this bundle, and neither original research repository is changed. See `PROVENANCE.md`.

## Why this stage exists

Lesson 9 used `ZAM_Paper-1_1_T-1.xml`, while our CARLA exercises used Town01. A Rulebook planner must receive road geometry aligned with the ego and obstacles. The current bridge reflects CARLA actor coordinates, but it does **not** create lanelets. This stage produces a **map-only** CommonRoad scenario and checks whether a sample of CARLA spawn points lands inside the converted lanelets. It does not yet create obstacle predictions, a planning problem, or execute `RPPSolver.plan()`.

## Copy files

Place `carla_map_export.py`, `convert_opendrive_map.py`, and `test_map_export.py` in:

```
D:\carla-rulebook\integration\
```

Keep the original Milestone 1 scripts in that directory. Put `README.md` / `PROVENANCE.md` in a separate subfolder such as `integration\milestone2_docs\` to avoid overwriting your existing research documentation.

## Step 1 — Test exporter without CARLA (PowerShell)

```
cd D:\carla-rulebook
.\.venv\Scripts\python.exe -m unittest discover -s integration -p "test_map_export.py" -v
```

Expected: four isolated tests pass. They use a fake CARLA map and do **not** test a live server, Scenario Designer, or map alignment.

## Step 2 — Export Town01 map (Windows PowerShell, when CARLA server is available)

Start **only your CARLA server** with Town01 loaded. You do **not** need a Scenic scenario or ego vehicle for a road map export. In another PowerShell terminal:

```
cd D:\carla-rulebook
.\.venv\Scripts\python.exe integration\carla_map_export.py --expect-map Town01
```

Produces (without overwriting existing files):

```
D:\carla-rulebook\integration\maps\Town01.xodr
D:\carla-rulebook\integration\maps\Town01.map_metadata.json
```

It uses `carla.Map.to_opendrive()` and `get_spawn_points()`. It never spawns actors, advances ticks, or controls vehicles.

**If CARLA isn't running now:** finish Step 1. If you already have a legitimate `Town01.xodr` from your CARLA installation, put it in `integration\maps\` and proceed to Step 3. Without exported CARLA metadata, the conversion still runs but skips the CARLA spawn-point alignment check. Do not invent metadata or reuse metadata from a different map. An `.xodr` file alone does not prove that its coordinates align with your active CARLA simulation.

## Step 3 — Install converter in a separate WSL environment

The current `commonroad-scenario-designer` package requires Python >=3.10. Our working `rpp_solver` environment uses Python 3.9; **do not install Scenario Designer into `rpp_solver`**.

In **WSL Ubuntu** (not PowerShell):

```
conda create -n cr_map python=3.11 -y
conda activate cr_map
python -m pip install "commonroad-scenario-designer==0.8.5"
python -c "from crdesigner.map_conversion.map_conversion_interface import opendrive_to_commonroad; print('Converter import OK')"
```

Installation and compatible library versions need verification on your machine. Pin `0.8.5` here for a reproducible initial attempt; record actual environment package versions in research notes.

## Step 4 — Convert the road map in WSL Ubuntu

```
conda activate cr_map
cd /mnt/d/carla-rulebook/integration
python convert_opendrive_map.py
```

You can override the paths using `--xodr`, `--metadata`, `--out`, and `--report`.

Expected products after conversion succeeds:

```
integration/maps/Town01.commonroad.xml
integration/maps/Town01.alignment_report.json
```

The XML is a **map-only** CommonRoad Scenario, written using CommonRoad-IO with an **empty** `PlanningProblemSet`. It does not yet represent a complete autonomous-driving experiment.

If `Town01.map_metadata.json` is available, the report checks that the XODR SHA-256 matches the export and counts how many exported CARLA spawn points lie in converted lanelets under (x, -y) and (x, y) conventions. It never automatically selects one. This is a **screening diagnostic**, not proof of exact alignment. Inspect the ratio and report before continuing. Some spawn-point mismatches can reflect coordinate-frame errors, map conversion choices, or missing lane geometry.

The converter refuses to overwrite existing output XML/reports; move them to an explicitly named archive if rerunning.

## Why no planner call yet?

To call the Rulebook planner correctly we still need: (1) verify actor-to-map registration using converted lanelets; (2) ego initial state / proper `PlanningProblem` and goal; (3) obstacle shapes and predictions; (4) reference path and planner configuration. We will not substitute Town01 coordinates into `ZAM_Paper-1_1_T-1.xml`, and we won't control CARLA until the planner has generated a verified trajectory on the matching map.

## Verified locally by assistant

- Python syntax checked for the two scripts and test file.
- `test_map_export.py`: four fake-map unit tests passed.
- No CARLA server, Scenario Designer installation, Town01 XODR, or `RPPSolver` was available in the assistant container; **no actual conversion or end-to-end map registration has been verified**.

## Source reference links

- CARLA 0.9.16 map documentation: https://carla.readthedocs.io/en/0.9.16/core_map/
- CARLA Map API: https://carla.readthedocs.io/en/latest/python_api/
- CommonRoad Scenario Designer code: https://github.com/CommonRoad/commonroad-scenario-designer
- OpenDRIVE → CommonRoad API: https://cps.pages.gitlab.lrz.de/commonroad/commonroad-scenario-designer/map_conversion_interface_api/
- Scenario Designer installation and Python versions: https://pypi.org/project/commonroad-scenario-designer/
