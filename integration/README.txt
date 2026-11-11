COMMONROAD/CARLA BRIDGE — MILESTONE 1 (read-only)
================================================
Location: D:\carla-rulebook\integration\  (Windows)
          /mnt/d/carla-rulebook/integration/ (WSL Ubuntu)

PURPOSE
  CARLA 0.9.16 (Windows Python 3.11) -> snapshot.json ->
  WSL rpp_solver (Python 3.9) -> CommonRoad InitialState and converted_snapshot.json.
  This validates state transport and coordinates. Does not invoke planning or control.
  No changes to CommonRoad or advisor's repositories.

SETUP: unzip files directly into D:\carla-rulebook\integration\

TEST WITHOUT CARLA:
  Windows PowerShell, in D:\carla-rulebook:
    .\.venv\Scripts\python.exe -m unittest discover -s integration -p "test_*.py" -v

WINDOWS TERMINAL A: Start your existing CARLA 0.9.16 server and whichever existing
agent/Scenic script spawns an ego vehicle. Keep the simulation running.

WINDOWS TERMINAL B (PowerShell):
    cd D:\carla-rulebook
    .\.venv\Scripts\python.exe integration\carla_export.py
  If multiple vehicles exist and none has role_name hero/ego_vehicle,
  the error lists all IDs. Select one explicitly:
    .\.venv\Scripts\python.exe integration\carla_export.py --ego-id 123
  Do not type "123" literally; use the printed CARLA vehicle ID.

WSL UBUNTU TERMINAL:
    conda activate rpp_solver
    cd /mnt/d/carla-rulebook/integration
    python commonroad_import.py

EXPECTED OUTPUT:
    integration/snapshot.json
    integration/converted_snapshot.json

IMPORTANT SCOPE
  The coordinate conversion is a mirror (x, -y, yaw -> -yaw), not a full
  CARLA-to-CommonRoad map conversion. It does NOT construct lanelets, dynamic
  obstacle predictions, planning problems, or trajectory following.
  In particular, ZAM_Paper-1_1_T-1.xml and CARLA Town01 are different maps.
  Do not call RPPSolver on a CARLA snapshot with an unmatched XML scenario.
  Pedestrians are captured as state records but not yet converted to proper
  CommonRoad obstacle shape/prediction objects.

NEXT
  Align road geometry/map and create CommonRoad Scenario + PlanningProblem
  around the CARLA state, then run one RPPSolver planning call, validate its
  output coordinates, and only then add controlled trajectory-following.
