"""JSON snapshot -> CommonRoad InitialState + planner-neutral scene JSON.

Run in WSL's existing `rpp_solver` conda env. This does NOT run RPPSolver yet:
the CARLA map must be converted/registered to a matching CommonRoad scenario.
"""
import argparse
import json
from pathlib import Path

from bridge_geometry import converted_actor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parent / 'snapshot.json')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'converted_snapshot.json')
    args = parser.parse_args()

    import numpy as np
    from commonroad.scenario.state import InitialState

    with args.input.open(encoding='utf-8') as f:
        raw = json.load(f)
    if raw.get('schema_version') != 1 or raw.get('source') != 'CARLA':
        raise ValueError('Unsupported snapshot schema/source')

    ego = converted_actor(raw['ego'])
    if ego['longitudinal_velocity_mps'] < -0.01:
        raise ValueError('Reverse motion: cannot represent this ego state with the basic forward-driving adapter')
    if abs(ego['lateral_velocity_mps']) > 0.5:
        print('WARNING: significant lateral ego velocity; InitialState does not encode it separately.')

    # This constructs a CommonRoad State; it is NOT a full CommonRoad Scenario.
    # time_step=0 is the relative first step of a future planning call, not CARLA frame number.
    cr_ego = InitialState(
        time_step=0,
        position=np.array(ego['position_m'], dtype=float),
        orientation=float(ego['orientation_rad']),
        velocity=max(0.0, float(ego['longitudinal_velocity_mps'])),
        acceleration=float(ego['longitudinal_acceleration_mps2']),
    )

    converted = {
        'schema_version': 1,
        'source_frame': raw['frame'],
        'source_elapsed_seconds': raw['elapsed_seconds'],
        'source_map': raw['map_name'],
        'coordinate_system': 'right-handed, CARLA X preserved, Y mirrored',
        'ego': ego,
        'obstacles': [converted_actor(v) for v in raw['obstacles']],
        'note': 'Geometric state conversion only; no lanelet map, prediction, or planner call yet.',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(converted, indent=2), encoding='utf-8')
    print(f"CommonRoad InitialState: position={cr_ego.position}, "
          f"orientation={cr_ego.orientation:.3f} rad, velocity={cr_ego.velocity:.3f} m/s")
    print(f"Converted {len(converted['obstacles'])} obstacle state(s). Output: {args.out}")
    print('NEXT MILESTONE: match CARLA map and obstacles to a CommonRoad scenario before RPPSolver.plan().')


if __name__ == '__main__':
    main()
