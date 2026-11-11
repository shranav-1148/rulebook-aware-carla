# Milestone 2A — provenance, external dependencies and validation

**Created:** November 10, 2026. **Nature:** New AI-assisted integration scaffolding for research inspection. **Status:** Not validated in live CARLA or Scenario Designer on the user's machine.

## Origins, API use, and code reuse

| File | Origin | Function | External references |
|---|---|---|---|
| `carla_map_export.py` | AI-generated helper written for this project | Calls `carla.Client`, `World.get_map`, `Map.to_opendrive`, `Map.get_spawn_points`; produces XODR and metadata | CARLA Python API and map documentation |
| `convert_opendrive_map.py` | AI-generated orchestration/diagnostic code for this project | Calls the *existing converter implementation* `crdesigner.map_conversion.map_conversion_interface.opendrive_to_commonroad`, CommonRoad-IO `CommonRoadFileWriter`, `PlanningProblemSet`, and `LaneletNetwork.find_lanelet_by_position` | CommonRoad Scenario Designer public API, CommonRoad-IO API |
| `test_map_export.py` | AI-generated tests with synthetic fake CARLA objects | Checks XML root/road detection, metadata, and filenames | Python standard-library unittest |
| `README.md` | AI-generated instructions for this specific WSL/Windows setup | Explicit usage and execution scope | Listed reference links |

**No blocks of third-party repository source are intentionally copied into these files.** Calling a documented API is not the same as independently implementing the algorithm behind that API. The actual OpenDRIVE-to-lanelet conversion algorithm is provided by CommonRoad Scenario Designer, not by the scripts in this bundle. These notes are not a guarantee of originality or a full similarity audit.

## Upstream references to record when publishing

- **CARLA 0.9.16 Map docs**: https://carla.readthedocs.io/en/0.9.16/core_map/
- **CARLA Python API**: https://carla.readthedocs.io/en/latest/python_api/
- **CommonRoad Scenario Designer**: https://github.com/CommonRoad/commonroad-scenario-designer — GPL-3.0 project. Check exact installed package's license and redistribution obligations if distributing software.
- **Scenario Designer API**: https://cps.pages.gitlab.lrz.de/commonroad/commonroad-scenario-designer/map_conversion_interface_api/ (`opendrive_to_commonroad`).
- **OpenDRIVE conversion principles**: M. Althoff, S. Urban, M. Koschi, “Automatic Conversion of Road Networks from OpenDRIVE to Lanelets,” IEEE SOLI 2018; background documentation: https://commonroad-scenario-designer.readthedocs.io/en/latest/details/open_drive/
- **CommonRoad-IO**: https://github.com/CommonRoad/commonroad-io — CommonRoad scene representation and XML writing.
- **Official CommonRoad–CARLA interface**: https://github.com/CommonRoad/commonroad-carla-interface — related infrastructure for comparison; **no source code copied from it**.
- **Advisor's planning repository**: https://github.com/tichakornw/planning — related research; **no source code copied from it**.

## Version and license policy

- Specify CARLA version, CARLA map name, CARLA server version, and XODR SHA-256 in exported metadata.
- Use a separate Python 3.11 `cr_map` Conda environment for Scenario Designer 0.8.5; preserve Python 3.9 `rpp_solver` unchanged.
- A script's use of a GPL-licensed dependency does not by itself settle the licensing status of distributable combined software. Review obligations with the research supervisor/institution before redistribution.
- We do not alter either upstream research repository.

## Tests and limitations

- Verified four unit tests, all synthetic and independent of a CARLA server or converter installation; compilation checks succeed.
- No live Town01 OpenDRIVE output available in this test environment. A successful XML conversion and lanelet count remain to be demonstrated.
- Spatial registration check uses sampled CARLA spawn points and `find_lanelet_by_position`. This is **not** sufficient to prove map alignment at all locations, correct lanes, or actual simulator obstacle safety.
- No scenario actors, dynamic obstacle predictions, destination/goal, reference path, Rulebook planning, CARLA control, or CARLA/Scenic timing synchronization are implemented by this milestone.

## File content hashes (SHA-256)

- `carla_map_export.py`: `a1c9c0c2b0ca06778ede827ad8e3705f0eb3260391ad3a3bad8fba37b618c61e`
- `convert_opendrive_map.py`: `81fcfa3e5c70c544978003eda7240df2f934022a63387919ae488f8f69151f9b`
- `test_map_export.py`: `b80bd8e842a1dbcec36495869fa8befde38e49bc9715d814992d6f1a825d3346`
- `README.md`: `e9c1bf57dae88b059324e5fab185a5090a200d5f08a0255c6ac70a109124eba8`
