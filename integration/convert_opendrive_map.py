"""Convert CARLA OpenDRIVE to a CommonRoad map using Scenario Designer's API.

Does not implement a map converter. Uses external (GPLv3) Scenario Designer;
source, license, and attribution in PROVENANCE.md. Run in WSL with a separate
Python >=3.10 environment containing commonroad-scenario-designer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert OpenDRIVE with official Scenario Designer')
    here = Path(__file__).resolve().parent
    parser.add_argument('--xodr', type=Path, default=here / 'maps' / 'Town01.xodr')
    parser.add_argument('--metadata', type=Path, default=here / 'maps' / 'Town01.map_metadata.json')
    parser.add_argument('--out', type=Path, default=here / 'maps' / 'Town01.commonroad.xml')
    parser.add_argument('--report', type=Path, default=here / 'maps' / 'Town01.alignment_report.json')
    args = parser.parse_args()

    if not args.xodr.is_file():
        raise FileNotFoundError(f'Missing OpenDRIVE input: {args.xodr}. Export or locate a real .xodr first.')
    for path in (args.out, args.report):
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite existing output {path}')

    source_hash = digest_file(args.xodr)
    source_data = None
    if args.metadata.is_file():
        source_data = json.loads(args.metadata.read_text(encoding='utf-8'))
        if source_data.get('schema_version') != 1 or source_data.get('source') != 'CARLA map API':
            raise ValueError('Unexpected CARLA map metadata format')
        if source_hash != source_data.get('opendrive_sha256'):
            raise ValueError('OpenDRIVE SHA-256 does not match accompanying metadata')
    else:
        print('WARNING: CARLA export metadata not found. Converting XODR without spawn-point alignment checks.')

    # External implementation: no copied converter algorithms in this file.
    import numpy as np
    from crdesigner.map_conversion.map_conversion_interface import opendrive_to_commonroad
    from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
    from commonroad.planning.planning_problem import PlanningProblemSet
    from commonroad.scenario.scenario import Tag

    print(f'Converting {args.xodr} with CommonRoad Scenario Designer ...')
    scenario = opendrive_to_commonroad(args.xodr)
    lanelet_network = scenario.lanelet_network
    lanelet_count = len(lanelet_network.lanelets)
    if lanelet_count == 0:
        raise RuntimeError('Conversion produced zero lanelets; no XML written')

    # Diagnostic, not proof of complete map agreement. Convert points using the
    # same mirror operation from our earlier bridge_geometry.py.
    samples = source_data.get('spawn_points_carla', []) if source_data else []
    mirrored = [np.array([float(s['x_m']), -float(s['y_m'])]) for s in samples]
    unmirrored = [np.array([float(s['x_m']), float(s['y_m'])]) for s in samples]

    def count_inside(points):
        if not points:
            return 0, []
        hits = lanelet_network.find_lanelet_by_position(points)
        return sum(bool(ids) for ids in hits), [int(bool(ids)) for ids in hits]

    mirrored_count, mirrored_hits = count_inside(mirrored)
    raw_count, raw_hits = count_inside(unmirrored)
    try:
        designer_version = version('commonroad-scenario-designer')
    except PackageNotFoundError:
        designer_version = 'unknown'
    report = {
        'carla_map_name': source_data['carla_map_name'] if source_data else None,
        'export_metadata_present': source_data is not None,
        'source_xodr_sha256': source_hash,
        'converter': 'commonroad-scenario-designer',
        'converter_version': designer_version,
        'commonroad_lanelet_count': lanelet_count,
        'spawn_points_checked': len(samples),
        'inside_lanelet_after_y_mirror': mirrored_count,
        'inside_lanelet_without_y_mirror': raw_count,
        'mirrored_hit_flags': mirrored_hits,
        'unmirrored_hit_flags': raw_hits,
        'interpretation': 'Diagnostics only; if metadata missing, no alignment evaluated. Even with hits, not proof of full map registration.',
    }

    writer = CommonRoadFileWriter(
        scenario=scenario,
        planning_problem_set=PlanningProblemSet(),
        author='CARLA/CommonRoad integration research prototype',
        affiliation='Independent research prototype',
        source='CARLA OpenDRIVE converted by CommonRoad Scenario Designer',
        tags={Tag.URBAN},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    writer.write_to_file(str(args.out), OverwriteExistingFile.ALWAYS)
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Converted lanelets: {lanelet_count}; output XML: {args.out}')
    print(f'CARLA spawn points within lanelets: mirrored={mirrored_count}/{len(samples)}, '
          f'raw={raw_count}/{len(samples)}')
    print(f'Alignment diagnostics: {args.report}')
    if samples and mirrored_count == 0:
        print('WARNING: no mirrored CARLA spawn points found within any converted lanelet.')
    print('No ego initial state, dynamic obstacles, planning goal, or Rulebook planner call created.')


if __name__ == '__main__':
    main()
