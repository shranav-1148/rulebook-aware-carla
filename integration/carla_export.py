"""Read-only CARLA -> JSON snapshot exporter, run on Windows Python 3.11.

No actor is spawned, controlled, destroyed, or ticked by this script.
"""
import argparse
import json
import os
from pathlib import Path


def actor_record(actor, state):
    t = state.get_transform()
    v = state.get_velocity()
    a = state.get_acceleration()
    bb = actor.bounding_box
    return {
        'actor_id': actor.id,
        'type_id': actor.type_id,
        'role_name': actor.attributes.get('role_name', ''),
        'pose_carla': {
            'x_m': t.location.x, 'y_m': t.location.y, 'z_m': t.location.z,
            'yaw_deg': t.rotation.yaw,
        },
        'velocity_carla_mps': {'x': v.x, 'y': v.y, 'z': v.z},
        'acceleration_carla_mps2': {'x': a.x, 'y': a.y, 'z': a.z},
        'bbox_carla_local': {
            'half_length_m': bb.extent.x, 'half_width_m': bb.extent.y,
            'half_height_m': bb.extent.z,
            'center_offset_m': {'x': bb.location.x, 'y': bb.location.y, 'z': bb.location.z},
        },
    }


def pick_ego(vehicles, ego_id):
    if ego_id is not None:
        matches = [v for v in vehicles if v.id == ego_id]
    else:
        matches = [v for v in vehicles if v.attributes.get('role_name') in ('hero', 'ego_vehicle')]
        if not matches and len(vehicles) == 1:
            matches = vehicles
    if len(matches) != 1:
        listing = '\n'.join(
            f"  id={v.id} type={v.type_id} role={v.attributes.get('role_name', '')}"
            for v in vehicles
        ) or '  (none)'
        raise RuntimeError(
            'Cannot identify a unique ego vehicle. Run your existing CARLA agent/Scenic '
            'scenario first, then use --ego-id ID if needed. Vehicles:\n' + listing
        )
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description='Export a single read-only CARLA snapshot.')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--ego-id', type=int, help='CARLA vehicle actor ID (optional if unambiguous)')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'snapshot.json')
    args = parser.parse_args()
    import carla  # only needed when actually connecting to CARLA

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    snapshot = world.get_snapshot()
    actors = world.get_actors()
    vehicles = list(actors.filter('vehicle.*'))
    walkers = list(actors.filter('walker.pedestrian.*'))
    ego = pick_ego(vehicles, args.ego_id)
    ego_snapshot = snapshot.find(ego.id)
    if ego_snapshot is None:
        raise RuntimeError('Ego actor is absent from the captured world frame; try again.')

    obstacles = []
    for actor in vehicles + walkers:
        if actor.id == ego.id:
            continue
        item_snapshot = snapshot.find(actor.id)
        if item_snapshot is not None:
            obstacles.append(actor_record(actor, item_snapshot))

    data = {
        'schema_version': 1,
        'source': 'CARLA',
        'frame': int(snapshot.frame),
        'elapsed_seconds': float(snapshot.timestamp.elapsed_seconds),
        'map_name': world.get_map().name,
        'coordinate_system': 'CARLA left-handed XY, yaw in degrees',
        'ego': actor_record(ego, ego_snapshot),
        'obstacles': obstacles,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, args.out)
    print(f"Saved frame {data['frame']} from {data['map_name']} -> {args.out}")
    print(f"Ego actor: {ego.id}; other vehicles/pedestrians: {len(obstacles)}")
    print('Read-only: did not tick CARLA or change vehicle control.')


if __name__ == '__main__':
    main()
