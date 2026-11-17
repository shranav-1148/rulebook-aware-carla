"""Milestone 6B-2S: passively capture a settled Town01 ego + stationary cone.

No motion commands, no world.tick, no synchronous-mode setting changes.
This add-on intentionally leaves Milestones 6B-1 and 6B-2 unchanged.
A stable snapshot is only an OFFLINE observation, not permission for control.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path

import carla_observe_m6b1 as old

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / 'output/carla_observation_6b2_stable/trial_001'


def finite(name, value):
    return old.finite(name, value)


def angle_difference_degrees(a, b):
    return abs(math.degrees(old.wrap(math.radians(a - b))))


@dataclass(frozen=True)
class StabilityCriteria:
    min_warmup_sim_s: float = 0.60  # relative to first paired observation, not spawn instant
    steady_dwell_sim_s: float = 0.45
    min_steady_samples: int = 8
    max_ego_horizontal_speed_mps: float = 0.05
    max_ego_vertical_speed_mps: float = 0.05
    max_prop_speed_mps: float = 0.02
    max_ego_position_drift_m: float = 0.05
    max_prop_position_drift_m: float = 0.02
    max_rotation_drift_deg: float = 1.0


def validate_criteria(c):
    for key, value in asdict(c).items():
        if key == 'min_steady_samples':
            if not isinstance(value, int) or value < 2:
                raise ValueError('min_steady_samples must be >=2')
        elif not math.isfinite(value) or value <= 0:
            raise ValueError(key + ' must be positive and finite')
    return c


def _actor_measurement(actor_snapshot):
    """Read pose and velocity only from an ActorSnapshot (published frame)."""
    pose = actor_snapshot.get_transform()
    vel = actor_snapshot.get_velocity()
    return {
        'xyz': tuple(finite('pose', x) for x in
                     (pose.location.x, pose.location.y, pose.location.z)),
        'rpy_deg': tuple(finite('rotation', x) for x in
                         (pose.rotation.roll, pose.rotation.pitch, pose.rotation.yaw)),
        'vxyz': tuple(finite('velocity', x) for x in (vel.x, vel.y, vel.z)),
    }


def extract_pair(snapshot, ego_id, obstacle_id):
    es = snapshot.find(ego_id)
    os = snapshot.find(obstacle_id)
    if es is None or os is None:
        return None
    return {
        'frame': int(snapshot.frame),
        'time_s': finite('snapshot elapsed seconds', snapshot.timestamp.elapsed_seconds),
        'ego': _actor_measurement(es),
        'obstacle': _actor_measurement(os),
    }


def distance_xyz(a, b):
    return math.sqrt(sum((x - y)**2 for x, y in zip(a, b)))


def rotation_drift(a, b):
    return max(angle_difference_degrees(x, y) for x, y in zip(a, b))


class PassiveStabilityGate:
    """Pure-data condition evaluator; it does not drive actors or own the CARLA world."""
    def __init__(self, criteria=None):
        self.criteria = validate_criteria(criteria or StabilityCriteria())
        self.first_time_s = None
        self.last_time_s = None
        self.last_frame = None
        self.frames_seen = 0
        self.missing_actor_frames = 0
        self.consecutive_stable = 0
        self.stable_begin_s = None
        self.ego_anchor = None
        self.obstacle_anchor = None
        self.latest_metrics = {}

    def _reset_stable(self):
        self.consecutive_stable = 0
        self.stable_begin_s = None
        self.ego_anchor = None
        self.obstacle_anchor = None

    def record_missing(self, frame):
        frame = int(frame)
        if self.last_frame is not None and frame <= self.last_frame:
            raise RuntimeError('CARLA snapshots are not strictly increasing')
        self.last_frame = frame
        self.frames_seen += 1
        self.missing_actor_frames += 1
        self._reset_stable()

    def update(self, pair):
        c = self.criteria
        t = finite('pair.time_s', pair['time_s'])
        f = int(pair['frame'])
        if self.last_frame is not None and f <= self.last_frame:
            raise RuntimeError('CARLA snapshots are not strictly increasing')
        if self.last_time_s is not None and t <= self.last_time_s:
            raise RuntimeError('CARLA simulation timestamps are not increasing')
        self.frames_seen += 1
        self.last_frame = f
        self.last_time_s = t
        if self.first_time_s is None:
            self.first_time_s = t
        age = t - self.first_time_s
        ego, prop = pair['ego'], pair['obstacle']
        vx, vy, vz = ego['vxyz']
        prop_speed = math.sqrt(sum(v*v for v in prop['vxyz']))
        horiz = math.hypot(vx, vy)
        speed_ok = horiz <= c.max_ego_horizontal_speed_mps and abs(vz) <= c.max_ego_vertical_speed_mps
        prop_ok = prop_speed <= c.max_prop_speed_mps
        warm = age >= c.min_warmup_sim_s
        ego_drift = prop_drift = rotation = 0.0
        if self.ego_anchor is not None:
            ego_drift = distance_xyz(ego['xyz'], self.ego_anchor['xyz'])
            prop_drift = distance_xyz(prop['xyz'], self.obstacle_anchor['xyz'])
            rotation = max(rotation_drift(ego['rpy_deg'], self.ego_anchor['rpy_deg']),
                           rotation_drift(prop['rpy_deg'], self.obstacle_anchor['rpy_deg']))
        still = (warm and speed_ok and prop_ok and
                 ego_drift <= c.max_ego_position_drift_m and
                 prop_drift <= c.max_prop_position_drift_m and
                 rotation <= c.max_rotation_drift_deg)
        if still:
            if self.stable_begin_s is None:
                self.stable_begin_s = t
                self.ego_anchor = ego
                self.obstacle_anchor = prop
            self.consecutive_stable += 1
        else:
            self._reset_stable()
        dwell = 0. if self.stable_begin_s is None else t - self.stable_begin_s
        accepted = (self.consecutive_stable >= c.min_steady_samples and
                    dwell >= c.steady_dwell_sim_s)
        self.latest_metrics = {
            'since_first_paired_frame_sim_s': age,
            'steady_dwell_sim_s': dwell,
            'steady_samples': self.consecutive_stable,
            'ego_horizontal_speed_mps': horiz,
            'ego_vertical_speed_mps': vz,
            'obstacle_3d_speed_mps': prop_speed,
            'ego_anchor_position_drift_m': ego_drift,
            'obstacle_anchor_position_drift_m': prop_drift,
            'max_anchor_rotation_drift_deg': rotation,
            'warmup_complete': warm,
            'instantaneous_speed_checks_passed': speed_ok and prop_ok,
            'accepted': accepted,
        }
        return accepted

    def summary(self):
        return {'criteria': asdict(self.criteria),
                'frames_seen': self.frames_seen,
                'missing_actor_frames': self.missing_actor_frames,
                'last_frame': self.last_frame,
                'first_paired_time_s': self.first_time_s,
                'last_time_s': self.last_time_s,
                'latest': self.latest_metrics,
                'certifies_ground_contact': False,
                'certifies_collision_mesh': False,
                'certifies_control_readiness': False}


def cone_origin_from_local_bbox(target_x, target_y, projected_waypoint_z,
                                bbox, *, clearance_m=0.02):
    """Place yaw-zero cone using actor-LOCAL bbox center/extents only.

    This intentionally avoids unpublished/stale post-spawn world transforms.
    The local bounding-box bottom is bbox.location.z - bbox.extent.z.
    Ground contact with off-road terrain is *not* established here.
    """
    coords = {
        'target_x': target_x, 'target_y': target_y,
        'projected_waypoint_z': projected_waypoint_z, 'clearance_m': clearance_m,
        'local_x': bbox.location.x, 'local_y': bbox.location.y,
        'local_z': bbox.location.z,
        'extent_x': bbox.extent.x, 'extent_y': bbox.extent.y,
        'extent_z': bbox.extent.z,
    }
    v = {name: finite(name, value) for name, value in coords.items()}
    if (min(v[k] for k in ('extent_x', 'extent_y', 'extent_z')) <= 0 or
        max(v[k] for k in ('extent_x', 'extent_y', 'extent_z')) > 1.0 or
        max(abs(v[k]) for k in ('local_x', 'local_y', 'local_z')) > 1.0):
        raise RuntimeError('Cone bounding box is not plausible actor-local geometry; '
                           'refusing to guess a placement transform')
    origin = (v['target_x'] - v['local_x'],
              v['target_y'] - v['local_y'],
              v['projected_waypoint_z'] + v['clearance_m']
              - (v['local_z'] - v['extent_z']))
    return origin, {
        'strategy': 'actor_local_bbox_direct_target_no_spawn_world_measurement',
        'requested_obstacle_center_carla_xy_m': [v['target_x'], v['target_y']],
        'target_origin_carla_xyz_m': list(origin),
        'bbox_local_center_xyz_m': [v['local_x'], v['local_y'], v['local_z']],
        'bbox_local_half_extents_xyz_m': [v['extent_x'], v['extent_y'], v['extent_z']],
        'projected_driving_waypoint_z_m': v['projected_waypoint_z'],
        'requested_bbox_bottom_z_m': v['projected_waypoint_z'] + v['clearance_m'],
        'ground_contact_verified': False,
    }


def wait_for_settled_pair(world, ego, prop, criteria, *, timeout_s=8.0,
                          max_frames=2500, max_elapsed_sim_s=8.0):
    gate = PassiveStabilityGate(criteria)
    last_snap = None
    for _ in range(max_frames):
        snap = world.wait_for_tick(timeout_s)  # passive in ALREADY asynchronous world
        if snap is None:
            raise RuntimeError('Timed out waiting for a CARLA world snapshot')
        last_snap = snap
        pair = extract_pair(snap, ego.id, prop.id)
        if pair is None:
            gate.record_missing(snap.frame)
            continue
        if gate.update(pair):
            return snap, gate.summary(), True
        if gate.first_time_s is not None and pair['time_s'] - gate.first_time_s >= max_elapsed_sim_s:
            break
    return last_snap, gate.summary(), False


def spawn_and_capture(args, ctx, carla, world, wp, server, blueprints):
    """Spawn only two owned actors; destroy them even on observation failure."""
    if not args.confirm_spawn or server['world_synchronous'] or server['existing_vehicle_count']:
        raise RuntimeError('Requires --confirm-spawn and an isolated asynchronous vehicle-free Town01')
    ego_bp, obstacle_bp = blueprints
    if not ego_bp.id.startswith('vehicle.') or not obstacle_bp.id.startswith('static.prop.'):
        raise RuntimeError('Unexpected ego/obstacle blueprint types')
    cx, cy = ctx['plan']['ego_initial_carla_body_center_xy_m']
    ox, oy = ctx['plan']['obstacle_model']['carla_xy_m']
    yaw = ctx['plan']['ego_initial_carla_yaw_deg']
    ego = prop = None
    observation = None
    stability = None
    gate_passed = False
    stability_gate_accepted = False
    post_settle_diagnostics = None
    failure = None
    cleanup = {}
    waypoint_z = None
    cone_geometry = None
    try:
        ego = world.try_spawn_actor(ego_bp, carla.Transform(
            carla.Location(cx, cy, wp.transform.location.z + args.ego_z_offset),
            carla.Rotation(yaw=yaw)))
        if ego is None:
            raise RuntimeError('Could not spawn temporary ego')
        origin_x, origin_y = old.actor_origin_for_body_center(cx, cy, yaw, ego.bounding_box.location)
        ego.set_transform(carla.Transform(
            carla.Location(origin_x, origin_y, wp.transform.location.z + args.ego_z_offset),
            carla.Rotation(yaw=yaw)))
        obs_wp = world.get_map().get_waypoint(
            carla.Location(ox, oy, 0), project_to_road=True, lane_type=carla.LaneType.Driving)
        if obs_wp is None:
            raise RuntimeError('Could not project cone to nearby driving waypoint')
        waypoint_z = float(obs_wp.transform.location.z)
        prop = world.try_spawn_actor(obstacle_bp, carla.Transform(
            carla.Location(ox, oy, waypoint_z + 3.0), carla.Rotation(yaw=0)))
        if prop is None:
            raise RuntimeError('Could not spawn temporary cone')
        # Compute the target ACTOR origin from its actor-local bbox only.
        # Do not use get_world_vertices(get_transform()) immediately after spawn:
        # an asynchronous server may not yet have published the prop's transform,
        # and subtracting a stale world center can inadvertently double XY.
        cone_origin, cone_geometry = cone_origin_from_local_bbox(
            ox, oy, waypoint_z, prop.bounding_box)
        prop.set_transform(carla.Transform(
            carla.Location(*cone_origin), carla.Rotation(yaw=0)))
        snap, stability, gate_passed = wait_for_settled_pair(
            world, ego, prop, StabilityCriteria(
                min_warmup_sim_s=args.warmup_s,
                steady_dwell_sim_s=args.dwell_s,
                min_steady_samples=args.min_samples),
            timeout_s=args.timeout, max_frames=args.max_frames,
            max_elapsed_sim_s=args.max_sim_seconds)
        stability_gate_accepted = gate_passed
        if gate_passed:
            observation = old.build_observation(snap, ego, prop, carla, ctx)
            required_checks = (
                'ego_pose_matches_nominal_start',
                'obstacle_center_matches_offline_location',
                'ego_within_route_offset_bound',
                'ego_within_route_heading_bound',
                'initial_wheel_steering_near_zero',
                'small_lateral_velocity',
            )
            observed_checks = {key: bool(observation['checks'][key]) for key in required_checks}
            failed_checks = [key for key, passed in observed_checks.items() if not passed]
            errors = observation['errors_vs_offline']
            ego_data = observation['ego']
            route_data = observation['ego_route_projection']
            post_settle_diagnostics = {
                'checks': observed_checks,
                'failed_checks': failed_checks,
                'snapshot_frame': observation['snapshot']['frame'],
                'ego_body_center_cr_xy_m': ego_data['body_center_cr_xy_m'],
                'expected_ego_body_center_cr_xy_m': ctx['plan']['ego_initial_commonroad_body_center_xy_m'],
                'ego_body_center_position_error_m': errors['ego_body_center_position_m'],
                'ego_heading_error_deg': errors['ego_heading_deg'],
                'ego_route_station_m': route_data['station_m'],
                'ego_route_offset_m': route_data['signed_lateral_offset_m'],
                'ego_route_heading_error_deg': abs(math.degrees(old.wrap(
                    ego_data['yaw_cr_rad'] - route_data['tangent_yaw_cr_rad']))),
                'ego_front_wheel_physics_angles_carla_deg': ego_data['front_wheel_physics_angles_carla_deg'],
                'ego_lateral_speed_mps': ego_data['lateral_speed_mps'],
                'obstacle_center_cr_xy_m': observation['obstacle']['center_cr_xy_m'],
                'expected_obstacle_center_cr_xy_m': ctx['plan']['obstacle_model']['commonroad_xy_m'],
                'obstacle_center_position_error_m': errors['obstacle_center_position_m'],
                'obstacle_actor_origin_carla_xyz_m': observation['obstacle']['actor_origin_carla_xyz_m'],
                'obstacle_bbox_local_center_xyz_m': observation['obstacle']['bbox']['local_center_xyz_m'],
                'obstacle_bbox_local_half_extents_m': observation['obstacle']['bbox']['local_half_extents_m'],
                'obstacle_observed_bbox_center_carla_xy_m': observation['obstacle']['center_carla_xy_m'],
            }
            if failed_checks:
                gate_passed = False
                failure = 'Post-settle checks failed: ' + ', '.join(failed_checks)
        else:
            failure = 'Vehicle/prop did not satisfy stable-frame gate within allotted sampling budget'
    except Exception as exc:
        failure = str(exc)
    finally:
        for label, actor in (('obstacle', prop), ('ego', ego)):
            if actor is not None:
                try:
                    cleanup[label] = {'actor_id': int(actor.id),
                                      'destroy_returned_success': bool(actor.destroy())}
                except Exception as exc:
                    cleanup[label] = {'actor_id': int(actor.id),
                                      'destroy_returned_success': False,
                                      'error': str(exc)}
        print('TEMPORARY ACTOR CLEANUP:', cleanup)
    cleanup_ok = (len(cleanup) == 2 and
                  all(entry['destroy_returned_success'] for entry in cleanup.values()))
    if not cleanup_ok:
        gate_passed = False
        failure = 'Temporary-actor destroy returned failure; inspect IDs before retrying'
    if stability is None:
        stability = {'frames_seen': 0}
    outcome = {'milestone': '6B-2S',
               'status': ('stable_snapshot_captured' if gate_passed else
                          'post_settle_validation_failed' if post_settle_diagnostics is not None and
                          post_settle_diagnostics['failed_checks'] else 'stability_capture_failed'),
               'stability': stability,
               'stability_gate_accepted': stability_gate_accepted,
               'post_settle_diagnostics': post_settle_diagnostics,
               'cleanup': cleanup,
               'cleanup_ok': cleanup_ok,
               'ground_height_reference': 'nearest projected driving waypoint, NOT off-road terrain',
               'obstacle_projected_waypoint_z_m': waypoint_z,
               'cone_placement': cone_geometry,
               'obstacle_ground_contact_verified': False,
               'control_applied': False,
               'autopilot_enabled': False,
               'client_world_ticks': 0,
               'physics_world_settings_modified': False,
               'live_control_ready': False,
               'failure_reason': failure}
    if observation is not None:
        outcome['snapshot_frame'] = observation['snapshot']['frame']
        outcome['cone_bbox_bottom_z_m'] = observation['obstacle']['bbox']['bounds_carla_xyz_m']['zmin']
        outcome['cone_bottom_minus_projected_waypoint_z_m'] = (
            outcome['cone_bbox_bottom_z_m'] - waypoint_z)
    if gate_passed:
        observation.update({'server': dict(server, actions_performed='two temporary actors spawned and destroyed'),
                            'observation_mode': 'temporary_spawn_passive_snapshot',
                            'stability_capture_mode': 'passive_multi_frame',
                            'stability': stability,
                            'owned_actor_ids': [ego.id, prop.id],
                            'ego_initial_velocity_seeded': False,
                            'client_world_ticks': 0,
                            'cleanup': cleanup,
                            'cleanup_ok': cleanup_ok,
                            'obstacle_ground_contact_verified': False,
                            'ready_for_live_control': False})
    return outcome, observation if gate_passed else None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-stable', action='store_true', required=True,
                   help='Explicitly request two temporary actors and passive settling capture')
    p.add_argument('--confirm-spawn', action='store_true', required=True)
    p.add_argument('--summary', type=Path, default=old.DEFAULT_EXPERIMENT/'summary.json')
    p.add_argument('--route', type=Path, default=old.DEFAULT_ROUTE)
    p.add_argument('--safety-csv', type=Path, default=old.DEFAULT_EXPERIMENT/'safety_first_bicycle_samples.csv')
    p.add_argument('--corridor-csv', type=Path, default=old.DEFAULT_EXPERIMENT/'corridor_first_bicycle_samples.csv')
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--ego-blueprint', default=old.DEFAULT_BLUEPRINT_EGO)
    p.add_argument('--obstacle-blueprint', default=old.DEFAULT_BLUEPRINT_PROP)
    p.add_argument('--ego-z-offset', type=float, default=0.5)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=2000)
    p.add_argument('--timeout', type=float, default=8.0)
    p.add_argument('--warmup-s', type=float, default=0.6)
    p.add_argument('--dwell-s', type=float, default=0.45)
    p.add_argument('--min-samples', type=int, default=8)
    p.add_argument('--max-sim-seconds', type=float, default=8.0)
    p.add_argument('--max-frames', type=int, default=2500)
    a = p.parse_args(argv)
    try:
        if not (a.capture_stable and a.confirm_spawn):
            raise ValueError('Both --capture-stable and --confirm-spawn are required')
        if any(not math.isfinite(v) or v <= 0 for v in
               (a.timeout, a.warmup_s, a.dwell_s, a.max_sim_seconds)):
            raise ValueError('Invalid timing values')
        if a.warmup_s + a.dwell_s >= a.max_sim_seconds:
            raise ValueError('Sampling duration too short for warmup + dwell')
        if not (0 <= a.ego_z_offset <= 1.5) or a.min_samples < 2 or a.max_frames < 2:
            raise ValueError('Invalid spawn offset or frame/sample limits')
        if not 1 <= a.port <= 65535:
            raise ValueError('Invalid port')
        if a.out.exists() and (a.out/'live_observation.json').exists():
            raise ValueError('Output already exists; use a fresh --out directory')
        if a.out.exists() and (a.out/'stable_capture_report.json').exists():
            raise ValueError('Output already exists; use a fresh --out directory')
        ctx = old.offline_context(a.summary, a.route, a.safety_csv, a.corridor_csv)
        carla, world, wp, server, blueprints = old.connect(a, ctx, need_blueprints=True)
        if server['world_synchronous'] or server['existing_vehicle_count']:
            raise RuntimeError('Refusing synchronous/vehicle-occupied server; use isolated async Town01')
        outcome, observation = spawn_and_capture(a, ctx, carla, world, wp, server, blueprints)
        a.out.mkdir(parents=True, exist_ok=True)
        report_path = old.save_json(a.out/'stable_capture_report.json', outcome)
        if observation is not None:
            observation_path = old.save_json(a.out/'live_observation.json', observation)
            print('STABLE OBSERVATION:', observation_path)
        else:
            print('No stable observation exported; do not use unstable source for planning')
            if outcome.get('post_settle_diagnostics'):
                print('Post-settle failed checks:', outcome['post_settle_diagnostics']['failed_checks'])
        print('MILESTONE 6B-2S:', outcome['status'])
        print('Stability report:', report_path)
        print('Actor cleanup returned success:', outcome['cleanup_ok'])
        print('No CARLA driving or client ticks issued.')
        return 0 if observation is not None else 3
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print('MILESTONE 6B-2S FAILED SAFE:', exc)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
