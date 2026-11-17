"""Milestone 6B-2: OFFLINE observed-state bicycle search (NO CARLA control).

This project-owned adapter deliberately does not modify Milestone 4E-B or 6A.
It reuses 4E-B's RK4 dynamics, geometry/route conventions, beam pruning,
lexicographic priority ranking, and trajectory serialization, while changing:
  * root state: measured CARLA observation, not route-station-zero constructor;
  * speed bound: explicitly 0..original max, so an observed parked vehicle is valid;
  * ego footprint: observed Model 3 bounding-box dimensions;
  * obstacle: measured 2D bbox polygon + an explicitly configured hard margin.

This is numerical OFFLINE exploration, NOT CARLA driving or certified safety.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

from shapely.geometry import MultiPoint, Point
from shapely.prepared import prep

# These are PROJECT-owned prior milestone modules, not external research packages.
import town01_bicycle_graph as m
from town01_feasibility_check import load_town01_road
from town01_motion_graph import read_route_csv

HERE = Path(__file__).resolve().parent
DEFAULT_OBSERVATION = HERE / 'output/carla_observation_6b1/live_observation.json'
DEFAULT_SUMMARY = HERE / 'output/town01_bicycle_graph_beam800_min40/summary.json'
DEFAULT_OUT = HERE / 'output/carla_plan_6b2'


def finite(name, value):
    if isinstance(value, bool):
        raise ValueError(f'{name} cannot be boolean')
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if not math.isfinite(x):
        raise ValueError(f'{name} must be finite')
    return x


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration_from_summary(summary, *, horizon_steps=None, beam_width=None):
    """Preserve exact 4E-B config, then explicitly permit nonnegative speed."""
    settings = dict(summary['settings'])
    for name in ('accelerations_mps2', 'steering_rates_deg_s'):
        settings[name] = tuple(settings[name])
    # Validates *original* config with its original positive speed lower bound.
    original = replace(m.Config(), **settings)
    m.validate_config(original)
    if horizon_steps is not None:
        if not isinstance(horizon_steps, int) or not 1 <= horizon_steps <= 12:
            raise ValueError('horizon_steps must be an integer in 1..12')
    if beam_width is not None:
        if not isinstance(beam_width, int) or not 1 <= beam_width <= 2000:
            raise ValueError('beam_width must be an integer in 1..2000')
    # 4E-B's original validate_config prohibits zero. The adapted primitive below
    # checks every sampled speed in [0, max], including the root state.
    return replace(original, speed_min_mps=0.0,
                   horizon_steps=horizon_steps or original.horizon_steps,
                   beam_width=beam_width or original.beam_width)


def _xy_vertices(observation, actor):
    vertices = observation[actor]['bbox']['vertices_commonroad_xy_m']
    if len(vertices) != 8:
        raise ValueError(actor + ' must provide exactly eight 3D bbox vertices')
    if any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in vertices):
        raise ValueError(actor + ' bounding box vertices must be XY pairs')
    return [(finite(actor + '.bbox.x', p[0]), finite(actor + '.bbox.y', p[1]))
            for p in vertices]


def measured_geometry(observation, cfg, *, obstacle_hard_margin_m=0.10):
    """Conservative 2D proxy, NOT an assertion about CARLA's collision mesh."""
    margin = finite('obstacle_hard_margin_m', obstacle_hard_margin_m)
    if not 0 <= margin <= 1:
        raise ValueError('Obstacle hard margin must be within 0..1 m')
    ego = observation['ego']['bbox']
    ex, ey, ez = (finite('ego half extent', x) for x in ego['local_half_extents_m'])
    if min(ex, ey, ez) <= 0:
        raise ValueError('Invalid ego bounding box extents')
    # Local box dimensions, not world axis-aligned widths (which rotate with yaw).
    cfg = replace(cfg, length_m=2.0 * ex, width_m=2.0 * ey)
    raw = MultiPoint(_xy_vertices(observation, 'obstacle')).convex_hull
    if raw.geom_type != 'Polygon' or raw.is_empty or raw.area <= 1e-6:
        raise ValueError('Measured obstacle footprint is degenerate')
    center = observation['obstacle']['center_cr_xy_m']
    if raw.centroid.distance(Point(finite('obstacle.cx', center[0]),
                                       finite('obstacle.cy', center[1]))) > .10:
        raise ValueError('Obstacle centroid and declared center disagree')
    hard = raw.buffer(margin)
    if hard.is_empty or not hard.is_valid:
        raise ValueError('Obstacle hard collision polygon is invalid')
    return cfg, raw, hard



def _stationary_negative_speed_to_zero(observation, speed):
    """Handle only signed negative residuals from VERIFIED passive settling.

    Return the raw value otherwise (including all unsupported reverse motion).
    This is an explicit modeling approximation, not a change to CARLA data.
    """
    if speed >= 0.0:
        return speed
    if observation.get('stability_capture_mode') != 'passive_multi_frame':
        return speed
    if observation.get('cleanup_ok') is not True:
        return speed
    stability = observation.get('stability') or {}
    latest = stability.get('latest') or {}
    criteria = stability.get('criteria') or {}
    if latest.get('accepted') is not True:
        return speed
    # Never exceed the capture's own horizontal/vertical settling thresholds.
    h_limit = min(0.05, finite('settling horizontal threshold',
                               criteria.get('max_ego_horizontal_speed_mps', 0)))
    v_limit = min(0.05, finite('settling vertical threshold',
                               criteria.get('max_ego_vertical_speed_mps', 0)))
    if h_limit <= 0 or v_limit <= 0:
        return speed
    velocity = observation['ego']['velocity_carla_xyz_mps']
    if len(velocity) != 3:
        raise ValueError('Expected three measured velocity components')
    vx, vy, vz = (finite('measured velocity', x) for x in velocity)
    horizontal = math.hypot(vx, vy)
    if (horizontal > h_limit or abs(vz) > v_limit or
            abs(speed) > h_limit or abs(speed) > horizontal + 1e-5):
        return speed
    # Reconfirm that the gate's accepted terminal frame was essentially at rest.
    if (finite('settling horizontal speed', latest.get('ego_horizontal_speed_mps')) > h_limit or
            abs(finite('settling vertical speed', latest.get('ego_vertical_speed_mps'))) > v_limit):
        return speed
    return 0.0


def observed_vehicle(observation, cfg, *, max_vertical_speed_mps=0.10):
    """Validate and return (4E-B Vehicle, issues). Stability issues are separate."""
    if observation.get('milestone') != '6B-1':
        raise ValueError('Expected Milestone 6B-1 observation')
    server = observation.get('server') or {}
    if server.get('map', '').split('/')[-1] != 'Town01':
        raise ValueError('Observation was not captured on Town01')
    if server.get('carla_client_version') != '0.9.16' or server.get('carla_server_version') != '0.9.16':
        raise ValueError('Unverified CARLA client/server version')
    if not str(observation['ego']['type_id']).startswith('vehicle.') or not str(observation['obstacle']['type_id']).startswith('static.prop.'):
        raise ValueError('Unexpected ego/obstacle types')
    if observation['ego']['actor_id'] == observation['obstacle']['actor_id']:
        raise ValueError('Ego and obstacle actor IDs must differ')
    if not isinstance(observation['snapshot']['frame'], int) or observation['snapshot']['frame'] < 0:
        raise ValueError('Invalid source frame')
    candidate = observation['planner_initial_state_candidate']
    q = m.Vehicle(*(finite(k, candidate[k]) for k in (
        'rear_axle_x_cr_m', 'rear_axle_y_cr_m', 'yaw_cr_rad', 'forward_speed_mps', 'steer_rad')))
    # Compare ORIGINAL candidate and measured speed before any interpretation.
    measured_speed = finite('observed forward speed', observation['ego']['longitudinal_speed_mps'])
    if abs(q.v - measured_speed) > .01:
        raise ValueError('Candidate speed differs from CARLA measured forward speed')
    # Verified parked captures may have signed negative residual velocity.
    # Only such near-zero residuals are represented as exact zero for a
    # forward-only bicycle model; CARLA observation data are not modified.
    q = replace(q, v=_stationary_negative_speed_to_zero(observation, q.v))
    if q.v < 0 or q.v > cfg.speed_max_mps + 1e-7:
        raise ValueError('Observed forward speed is outside the supported [0, max] bicycle range')
    if abs(q.steer) > m.rad(cfg.max_abs_steering_deg) + 1e-7:
        raise ValueError('Observed steering exceeds planner limit')
    angles = observation['ego']['front_wheel_physics_angles_carla_deg']
    if len(angles) != 2 or max(abs(finite('wheel angle', a)) for a in angles) >= .5 or abs(q.steer) > 1e-8:
        raise ValueError('Only near-zero observed wheel steering is supported until sign calibration')
    cx, cy = m.body_center(q, cfg)
    obs_cx, obs_cy = (finite('ego center', v) for v in observation['ego']['body_center_cr_xy_m'])
    if math.hypot(cx - obs_cx, cy - obs_cy) > .05:
        raise ValueError('Observed rear axle and measured ego body center are inconsistent')
    lateral = finite('lateral velocity', observation['ego']['lateral_speed_mps'])
    if abs(lateral) > .5:
        raise ValueError('Significant lateral velocity violates the no-slip bicycle assumption')
    vz = finite('vertical velocity', observation['ego']['velocity_carla_xyz_mps'][2])
    issues = []
    if abs(vz) > max_vertical_speed_mps:
        issues.append(f'ego is not vertically settled (|vz|={abs(vz):.3f} m/s)')
    if observation.get('observation_mode') == 'temporary_spawn_passive_snapshot' and observation.get('cleanup_ok') is not True:
        issues.append('temporary actor cleanup was not confirmed')
    if observation['obstacle'].get('collision_mesh_verified') is not True:
        issues.append('CARLA collision mesh remains unverified; only bbox proxy is available')
    return q, issues


def evaluate(q, cfg, route, road, prepared, obstacle_hard):
    """Feasibility & priority costs at ONE integration sample."""
    if q.v < -1e-7 or q.v > cfg.speed_max_mps + 1e-7:
        return None, 'speed_limit'
    if abs(q.steer) > m.rad(cfg.max_abs_steering_deg) + 1e-7:
        return None, 'steering_limit'
    if q.v * q.v * abs(math.tan(q.steer)) / cfg.wheelbase_m > cfg.max_lateral_accel_mps2 + 1e-7:
        return None, 'lateral_accel_limit'
    cx, cy = m.body_center(q, cfg)
    s, d, tangent = m.route_projection(route, cx, cy)
    heading_error = m.wrap(q.yaw - tangent)
    if abs(d) > cfg.max_abs_route_offset_m + 1e-7 or abs(heading_error) > m.rad(cfg.max_abs_heading_error_deg) + 1e-7:
        return None, 'route_deviation_limit'
    footprint = m._footprint(q, cfg)
    if not prepared.covers(footprint) and footprint.difference(road).area > cfg.road_area_tolerance_m2:
        return None, 'road_footprint'
    if footprint.intersects(obstacle_hard):
        return None, 'hard_obstacle_collision'
    clearance = footprint.distance(obstacle_hard)
    return (s, d, heading_error, max(0., cfg.clearance_margin_m - clearance), d*d), None


def observed_primitive(parent, acceleration, steering_rate, cfg, route, road, prepared, hard):
    if abs(parent.state.steer + m.rad(steering_rate) * cfg.dt_s) > m.rad(cfg.max_abs_steering_deg) + 1e-9:
        return None, 'steering_limit'
    count = max(2, math.ceil(cfg.dt_s / cfg.integration_dt_s))
    dt = cfg.dt_s / count
    current = parent.state
    prev, reason = evaluate(current, cfg, route, road, prepared, hard)
    if reason:
        return None, 'invalid_source_' + reason
    samples, r9, r5 = [], 0., 0.
    for i in range(1, count + 1):
        current = m.rk4_step(current, acceleration, steering_rate, dt, cfg.wheelbase_m)
        cur, reason = evaluate(current, cfg, route, road, prepared, hard)
        if reason:
            return None, reason
        r9 += dt * (prev[3] + cur[3]) / 2
        r5 += dt * (prev[4] + cur[4]) / 2
        samples.append((i * dt, current, cur[0], cur[1], cur[2]))
        prev = cur
    r4 = cfg.dt_s * (acceleration * acceleration + (steering_rate / 8.)**2)
    return (samples, (r9, r5, r4), prev[0], prev[1], prev[2]), None


def build_observed_motion_tree(route, road, cfg, start, hard):
    """Preserve original search/ranking, but use the measured root and obstacle."""
    if not road.is_valid or road.is_empty:
        raise ValueError('Road geometry invalid')
    prepared = prep(road)
    first, reason = evaluate(start, cfg, route, road, prepared, hard)
    if reason:
        raise ValueError('Observed initial state inadmissible: ' + reason)
    s, d, heading = first[:3]
    root = m.Node(0, 0, start, None, None, s, d, heading, (0., 0., 0.), [])
    nodes, layer, layers, rejects, next_id = {0: root}, [root], [], Counter(), 1
    for step in range(cfg.horizon_steps):
        candidates = []
        for src in layer:
            for acceleration in cfg.accelerations_mps2:
                for steering_rate in cfg.steering_rates_deg_s:
                    outcome, reason = observed_primitive(src, acceleration, steering_rate, cfg, route, road, prepared, hard)
                    if reason:
                        rejects[reason] += 1
                        continue
                    samples, cost, sn, dn, psi = outcome
                    end = samples[-1][1]
                    total = tuple(src.costs[j] + cost[j] for j in range(3))
                    candidates.append(m.Node(next_id, step+1, end, src.node_id,
                                             (acceleration, steering_rate), sn, dn,
                                             psi, total, samples))
                    next_id += 1
        if not candidates:
            raise RuntimeError(f'No feasible observed-state primitives at step {step+1}; rejections={dict(rejects)}')
        chosen = m.select_diverse_candidates(candidates, cfg)
        for node in chosen:
            nodes[node.node_id] = node
        layers.append({'step': step+1, 'parents': len(layer),
                       'viable_candidate_edges': len(candidates),
                       'retained_states': len(chosen),
                       'pruned_candidates': len(candidates)-len(chosen)})
        layer = chosen
    return nodes, layer, layers, dict(rejects)


def diagnostics(path, road, hard, cfg):
    states = [path[0].state] + [sample[1] for node in path[1:] for sample in node.samples]
    outside = max(m.footprint_outside_area(q, road, cfg) for q in states)
    gap = min(m._footprint(q, cfg).distance(hard) for q in states)
    intersects = any(m._footprint(q, cfg).intersects(hard) for q in states)
    return {'sample_count': len(states), 'maximum_road_outside_area_m2': outside,
            'minimum_bbox_proxy_clearance_after_margin_m': gap,
            'sampled_road_pass': outside <= cfg.road_area_tolerance_m2 + 1e-8,
            'sampled_hard_obstacle_pass': not intersects,
            'sampled_max_speed_mps': max(q.v for q in states),
            'sampled_min_speed_mps': min(q.v for q in states),
            'sampled_max_abs_steering_deg': max(abs(math.degrees(q.steer)) for q in states)}


def check_provenance(summary, route_path, map_path):
    if summary.get('route_sha256') != sha256(route_path):
        raise ValueError('Route SHA256 mismatch with the recorded 4E-B experiment')
    if summary.get('map_sha256') != sha256(map_path):
        raise ValueError('Map SHA256 mismatch with the recorded 4E-B experiment')
    if summary.get('experiment', '').startswith('Milestone 4E-B') is False:
        raise ValueError('Expected a Milestone 4E-B summary')


def verify_observation_scenario(observation, summary):
    '''Do not silently use a 6B-1 observation from another offline scenario.'''
    center = observation['offline_obstacle']['commonroad_xy_m']
    source = summary['obstacle_xy_m']
    if len(center) != 2 or len(source) != 2:
        raise ValueError('Missing source obstacle center')
    distance = math.hypot(finite('observed reference obstacle x', center[0])-finite('summary obstacle x', source[0]),
                          finite('observed reference obstacle y', center[1])-finite('summary obstacle y', source[1]))
    if distance > 0.02:
        raise ValueError('6B-1 source obstacle differs from 4E-B summary; mixed experiments')


def run_offline(observation, summary, route, road, *, min_progress_m=5.0,
                hard_margin_m=.10, horizon_steps=None, beam_width=None,
                allow_unstable_offline=False):
    cfg = configuration_from_summary(summary, horizon_steps=horizon_steps, beam_width=beam_width)
    cfg, raw_obstacle, hard_obstacle = measured_geometry(observation, cfg,
                                                        obstacle_hard_margin_m=hard_margin_m)
    start, issues = observed_vehicle(observation, cfg)
    initial_s, initial_d, initial_tangent = m.route_projection(route, *m.body_center(start, cfg))
    progress = finite('minimum progress', min_progress_m)
    if progress < 0:
        raise ValueError('Minimum progress cannot be negative')
    min_final_s = initial_s + progress
    if min_final_s > route[-1][0]:
        raise ValueError('Requested progress exceeds route end')
    result = {'milestone': '6B-2', 'mode': 'offline_observation_replay_no_carla',
              'source_frame': observation['snapshot']['frame'],
              'start_state': asdict(start),
              'raw_observed_forward_speed_mps': finite('observed forward speed', observation['ego']['longitudinal_speed_mps']),
              'stationary_negative_speed_normalized_to_zero': (
                  finite('observed forward speed', observation['ego']['longitudinal_speed_mps']) < 0 and start.v == 0.0),
              'start_station_m': initial_s,
              'start_offset_m': initial_d,
              'minimum_final_station_m': min_final_s,
              'settings': asdict(cfg),
              'measured_ego_bbox_length_width_m': [cfg.length_m, cfg.width_m],
              'measured_obstacle_footprint_area_m2': raw_obstacle.area,
              'measured_obstacle_bbox_cr_xy_exterior_m': [list(p) for p in raw_obstacle.exterior.coords],
              'hard_obstacle_margin_m': hard_margin_m,
              'stability_and_model_issues': issues,
              'executed_carla_control': False,
              'ready_for_live_control': False,
              'limitations': ['One snapshot cannot certify settled vehicle dynamics',
                              'CARLA collision mesh unverified; uses XY bounding-box proxy',
                              '2.7m wheelbase and rear-axle offset remain assumptions',
                              'Uncalibrated nonzero steering is rejected',
                              'No control actuation/feedback/tire/engine model',
                              'Only integration-sampled collision checks, not continuous guarantees',
                              'Beam-pruned lexicographic proxy ranking is NOT the original Rulebook']}
    # We distinguish physical settling from the collision mesh uncertainty:
    # unverified mesh does NOT block offline bbox-proxy search, but does block
    # treating it as live-controllable or physical collision certified.
    unstable = [i for i in issues if 'vertically settled' in i or 'cleanup' in i]
    if unstable and not allow_unstable_offline:
        result.update(status='blocked_unstable_observation',
                      explanation='Use a physically settled observation. --allow-unstable-offline enables numerical replay ONLY, never driving.')
        return result, None
    if unstable:
        result['offline_unstable_override_used'] = True
    try:
        nodes, terminal, layers, rejections = build_observed_motion_tree(route, road, cfg, start, hard_obstacle)
        selected = {}
        for profile in ('safety_first', 'corridor_first'):
            path = m.select_path(nodes, terminal, profile, min_final_s)
            dense = m.path_samples(path, cfg, profile, route)
            diag = diagnostics(path, road, hard_obstacle, cfg)
            if not (diag['sampled_road_pass'] and diag['sampled_hard_obstacle_pass']):
                raise AssertionError('Selected trajectory failed sampled geometry validation')
            selected[profile] = {'path': path, 'dense': dense, 'diagnostic': diag}
    except (ValueError, RuntimeError) as exc:
        result.update(status='offline_planning_failed', explanation=str(exc))
        return result, None
    result.update(status='offline_plan_generated_unstable_input' if unstable else 'offline_plan_generated',
                  shared_tree=True, tree_nodes_retained=len(nodes), layers=layers,
                  rejections_by_reason=rejections,
                  profiles={name: {'costs_R9_R5_R4': list(data['path'][-1].costs),
                                   'final_station_m': data['path'][-1].station,
                                   'diagnostic': data['diagnostic'],
                                   'actions': [{'acceleration_mps2': n.action[0],
                                                'steering_rate_deg_s': n.action[1]}
                                               for n in data['path'][1:]]}
                            for name, data in selected.items()},
                  divergence=m.compare_paths(selected['safety_first']['dense'],
                                             selected['corridor_first']['dense']))
    return result, selected


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observation', type=Path, default=DEFAULT_OBSERVATION)
    p.add_argument('--summary', type=Path, default=DEFAULT_SUMMARY)
    p.add_argument('--route', type=Path, default=m.DEFAULT_ROUTE)
    p.add_argument('--map', type=Path, default=m.DEFAULT_MAP)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--min-progress-m', type=float, default=5.0)
    p.add_argument('--hard-margin-m', type=float, default=.10)
    p.add_argument('--horizon-steps', type=int)
    p.add_argument('--beam-width', type=int)
    p.add_argument('--allow-unstable-offline', action='store_true',
                   help='ONLY for mathematical replay of falling spawn; does NOT make state physically valid')
    a = p.parse_args(argv)
    try:
        if a.out.joinpath('report.json').exists():
            raise ValueError('Output report already exists; choose a new --out folder to preserve earlier results')
        observation = json.loads(a.observation.read_text(encoding='utf-8'))
        summary = json.loads(a.summary.read_text(encoding='utf-8'))
        check_provenance(summary, a.route, a.map)
        verify_observation_scenario(observation, summary)
        route = read_route_csv(a.route)
        road, lanelet_count = load_town01_road(a.map, m.VALIDATION_LANELETS)
        report, selected = run_offline(observation, summary, route, road,
                                      min_progress_m=a.min_progress_m,
                                      hard_margin_m=a.hard_margin_m,
                                      horizon_steps=a.horizon_steps,
                                      beam_width=a.beam_width,
                                      allow_unstable_offline=a.allow_unstable_offline)
        report['source_provenance'] = {'observation_sha256': sha256(a.observation),
                                       'summary_sha256': sha256(a.summary),
                                       'route_sha256': sha256(a.route),
                                       'map_sha256': sha256(a.map),
                                       'validation_lanelet_count': lanelet_count}
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        if selected:
            for name, entry in selected.items():
                m.save_csv(a.out / (name + '_observed_samples.csv'), entry['dense'])
        print('MILESTONE 6B-2:', report['status'])
        print('Observed forward speed (raw):', report['raw_observed_forward_speed_mps'], 'm/s')
        print('Planner initial forward speed:', report['start_state']['v'], 'm/s')
        print('Stationary negative-speed normalization:',
              report['stationary_negative_speed_normalized_to_zero'])
        print('Ego geometry (length,width):', report['measured_ego_bbox_length_width_m'])
        print('Issues:', report['stability_and_model_issues'])
        print('No CARLA control applied. Output:', a.out/'report.json')
        return 0 if selected else 3
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print('MILESTONE 6B-2 FAILED SAFE:', exc)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
