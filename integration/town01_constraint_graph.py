"""Milestone 4E-A: Reject road-footprint-invalid edges BEFORE Rulebook selection.

Standalone project-authored integration experiment. Reuses the existing M4B
state graph and R9/R5 proxy costs, and M4D's geometry/dynamics diagnostics.
Does NOT implement a bicycle model or guarantee continuous-time road safety.
Does NOT modify any upstream or prior milestone source files.
"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

from shapely.geometry import Polygon
from shapely.prepared import prep

from town01_motion_graph import (
    Settings, build_lattice, divergence, edge_geometry_and_cost,
    initial_state, maybe_plot, next_state, path_rows, pose_at,
    read_route_csv, save_csv, select_trajectory, synthetic_route,
    unpack_state, validate_settings, world_xy,
)
from town01_feasibility_check import (
    Limits, inferred_yaw, load_town01_road, plot_report,
    validate_trajectory, vehicle_rectangle, verify_limits,
)

BASE = Path(__file__).resolve().parent
MAP_PATH = BASE / 'maps' / 'Town01.rpp_compat.xml'
ROUTE_PATH = BASE / 'output' / 'town01_graph' / 'route_waypoints.csv'
OUT_PATH = BASE / 'output' / 'town01_motion_graph_constrained'
PLANNED_ROUTE_IDS = (38, 40, 58, 60)
# The measured ego center starts at the beginning of 38, and the rear overlaps 39.
VALIDATION_ROAD_IDS = (39, 38, 40, 58, 60)


def road_footprint_at(route, s, d, v, vy, limits):
    """Approximate footprint: heading inferred from route tangent and velocity."""
    x, y = world_xy(route, s, d)
    yaw = inferred_yaw(route, s, v, vy)
    return vehicle_rectangle(x, y, yaw, limits.length_m, limits.width_m)


def edge_road_violation(state, nxt, ax, ay, cfg, route, road, limits, prepared=None):
    """Return None, 'quantization', or 'road_footprint' for an edge.

    Include actual (unquantized) constant-acceleration path, both endpoints,
    AND quantized endpoint. This tests *sampled* occupancy, not a swept solid.
    """
    _, s, d, v, vy = unpack_state(state, cfg)
    _, sn, dn, vn, vyn = unpack_state(nxt, cfg)
    dt = cfg.dt
    expected_s = s + v * dt + 0.5 * ax * dt * dt
    expected_d = d + vy * dt + 0.5 * ay * dt * dt
    expected_v = v + ax * dt
    expected_vy = vy + ay * dt
    if math.hypot(expected_s - sn, expected_d - dn) > limits.max_quantization_jump_m + 1e-9:
        return 'quantization'
    if math.hypot(expected_v - vn, expected_vy - vyn) > 1e-8:
        return 'quantization'
    n = max(2, math.ceil(dt / limits.sample_dt_s))
    prepared = prepared if prepared is not None else prep(road)
    for j in range(n + 1):
        tau = dt * j / n
        station = s + v * tau + 0.5 * ax * tau * tau
        lat = d + vy * tau + 0.5 * ay * tau * tau
        vt = v + ax * tau
        vyt = vy + ay * tau
        footprint = road_footprint_at(route, station, lat, vt, vyt, limits)
        if not prepared.covers(footprint):
            # The same area-tolerance convention used by M4D.
            if footprint.difference(road).area > limits.area_tolerance_m2:
                return 'road_footprint'
    # Check the actual snapped state as well, not just the ideal edge endpoint.
    footprint = road_footprint_at(route, sn, dn, vn, vyn, limits)
    if not prepared.covers(footprint) and footprint.difference(road).area > limits.area_tolerance_m2:
        return 'road_footprint'
    return None


def sampled_edge_road_ok(state, nxt, ax, ay, cfg, route, road, limits, prepared=None):
    return edge_road_violation(state, nxt, ax, ay, cfg, route, road, limits, prepared) is None


def build_constrained_lattice(route, cfg, road, limits):
    """Build one shared DAG; filter each candidate edge on road-footprint samples.

    Returns same leading five values as M4B build_lattice plus rejection stats.
    The original M4B graph/cost functions are NOT modified.
    """
    validate_settings(cfg, route)
    verify_limits(limits)
    if road.is_empty or not road.is_valid:
        raise ValueError('Invalid road polygon; repair/review source geometry before planning')
    prepared = prep(road)
    start = initial_state(cfg)
    _, s0, d0, v0, vy0 = unpack_state(start, cfg)
    first_footprint = road_footprint_at(route, s0, d0, v0, vy0, limits)
    if not prepared.covers(first_footprint) and first_footprint.difference(road).area > limits.area_tolerance_m2:
        raise ValueError('Initial ego footprint outside configured validation road region')
    obstacle = world_xy(route, cfg.obstacle_station, cfg.obstacle_offset)
    layer = {start}
    edges = {}
    counts = []
    failures = Counter()
    for step in range(cfg.horizon):
        next_layer = set()
        edges_accepted = 0
        for state in sorted(layer):
            outbound = []
            for ax in cfg.ax_actions:
                for ay in cfg.ay_actions:
                    nxt = next_state(state, ax, ay, cfg, route[-1][0])
                    if nxt is None:
                        failures['lattice_or_assumed_corridor'] += 1
                        continue
                    costs = edge_geometry_and_cost(state, nxt, ax, ay, cfg, route, obstacle)
                    if costs is None:
                        failures['proxy_obstacle_collision_or_offset'] += 1
                        continue
                    reason = edge_road_violation(state, nxt, ax, ay, cfg, route, road, limits, prepared)
                    if reason is not None:
                        failures[reason] += 1
                        continue
                    outbound.append((nxt, costs, (ax, ay)))
                    next_layer.add(nxt)
                    edges_accepted += 1
            edges[state] = outbound
        counts.append({'time_index': step + 1,
                       'states': len(next_layer),
                       'edges_from_previous': edges_accepted})
        layer = next_layer
        if not layer:
            raise RuntimeError('Constrained graph has no reachable states at step {}. '
                               'Review road geometry, obstacle, or action constraints.'.format(step + 1))
    return start, edges, sorted(layer), counts, obstacle, dict(failures)


def edge_set(graph):
    return {(src, nxt, tuple(action))
            for src, outs in graph.items() for nxt, _, action in outs}


def experiment(route, road, cfg, limits, min_final_s=0.0, baseline=True):
    """Return summary and rows. All profiles share exactly ONE constrained graph."""
    start, graph, final_states, layers, obstacle, rejections = build_constrained_lattice(
        route, cfg, road, limits)
    eligible = [st for st in final_states if unpack_state(st, cfg)[1] >= min_final_s - 1e-9]
    if not eligible:
        raise RuntimeError('No constrained terminal state satisfies min-final-s={}'.format(min_final_s))
    saved = {}
    for profile in ('safety_first', 'corridor_first'):
        states, actions, costs = select_trajectory(start, graph, eligible, profile, cfg)
        rows = path_rows(states, actions, costs, profile, cfg, route)
        diagnostic, dense, seams = validate_trajectory(rows, route, road, limits)
        if not diagnostic['road_pass']:
            raise AssertionError('Internal mismatch: selected trajectory fails M4D road sampling')
        saved[profile] = {'rows': rows, 'costs': costs, 'diagnostic': diagnostic,
                          'dense': dense, 'seams': seams, 'states': states}
    base = None
    if baseline:
        b_start, b_graph, b_terminals, b_layers, _ = build_lattice(route, cfg)
        if not edge_set(graph).issubset(edge_set(b_graph)):
            raise AssertionError('Constrained graph is not a subset of baseline edge set')
        b_eligible = [st for st in b_terminals if unpack_state(st, cfg)[1] >= min_final_s - 1e-9]
        base = {'states': 1 + sum(x['states'] for x in b_layers),
                'edges': sum(x['edges_from_previous'] for x in b_layers)}
        if b_eligible:
            base['selection'] = {}
            for profile in ('safety_first', 'corridor_first'):
                b_states, _, b_costs = select_trajectory(b_start, b_graph, b_eligible, profile, cfg)
                base['selection'][profile] = {
                    'costs': {'R9_proxy': b_costs[0], 'R5_proxy': b_costs[1], 'R4_proxy': b_costs[2]},
                    'changed_by_road_constraint': b_states != saved[profile]['states']}
    report = {
        'experiment': 'M4E-A: sampled road-footprint edge pruning before lexicographic priority selection',
        'graph': {'time_layered_states': 1 + sum(x['states'] for x in layers),
                  'edges': sum(x['edges_from_previous'] for x in layers),
                  'layers': layers, 'edge_rejection_attempt_counts': rejections,
                  'one_shared_feasible_graph_for_both_priorities': True},
        'assumed_vehicle': asdict(limits),
        'motion_graph_settings': asdict(cfg),
        'minimum_terminal_progress_m': min_final_s,
        'baseline_unconstrained_graph': base,
        'profiles': {p: {
            'costs': {'R9_proxy': saved[p]['costs'][0],
                      'R5_proxy': saved[p]['costs'][1],
                      'R4_proxy': saved[p]['costs'][2]},
            'last_station_m': saved[p]['rows'][-1]['s_m'],
            'feasibility_diagnostic': saved[p]['diagnostic']
        } for p in saved},
        'divergence': divergence(saved['safety_first']['rows'], saved['corridor_first']['rows']),
        'warnings': [
            'Road containment checked at discrete <= sample_dt_s points per edge, NOT as exact continuous swept geometry.',
            'Yaw inferred from route tangent and velocity, not integrated from vehicle steering dynamics.',
            'No bicycle-model steering/steering-rate hard constraint in graph generation yet (next stage).',
            'Obstacle collision still uses original M4B sampled circle proxy, not CARLA actor footprint.',
            'R9/R5/R4 are educational proxy costs and lexicographic TOTAL orders, not original research Rulebook.',
            'Vehicle size and road-admissibility assumptions are not calibrated to actual CARLA ego vehicle.',
            'Compatibility XML replaced curb labels with unknown; road-rule semantics unverified.',
            'Offline simulation only; no Scenic scenario or CARLA vehicle execution.'
        ]}
    return report, saved, obstacle


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--map', type=Path, default=MAP_PATH)
    ap.add_argument('--route', type=Path, default=ROUTE_PATH)
    ap.add_argument('--out', type=Path, default=OUT_PATH)
    ap.add_argument('--demo', action='store_true', help='Use an illustrative synthetic road; no CommonRoad required')
    ap.add_argument('--sample-dt', type=float, default=0.1)
    ap.add_argument('--vehicle-length', type=float, default=4.5)
    ap.add_argument('--vehicle-width', type=float, default=1.9)
    ap.add_argument('--min-final-s', type=float, default=0.0)
    ap.add_argument('--no-baseline', action='store_true', help='Skip comparison with unfiltered M4B graph')
    args = ap.parse_args()
    if not math.isfinite(args.min_final_s) or args.min_final_s < 0:
        ap.error('min-final-s must be finite and nonnegative')
    limits = replace(Limits(), length_m=args.vehicle_length, width_m=args.vehicle_width,
                     sample_dt_s=args.sample_dt)
    verify_limits(limits)
    cfg = Settings()
    if args.demo:
        route = synthetic_route()
        road = Polygon([(-4,-1.75),(84,-1.75),(84,1.75),(-4,1.75)])
        n_lanelets = None
    else:
        if not args.map.is_file() or not args.route.is_file():
            ap.error('Missing Town01 map or route. Run Milestone 4A and ensure map XML exists.')
        route = read_route_csv(args.route)
        road, n_lanelets = load_town01_road(args.map, ids=VALIDATION_ROAD_IDS)
    print('[4E] Building the constrained graph and testing full vehicle footprints on each candidate edge...')
    report, results, obstacle = experiment(route, road, cfg, limits,
                                          min_final_s=args.min_final_s,
                                          baseline=not args.no_baseline)
    report['route_source'] = 'synthetic-demo' if args.demo else str(args.route)
    report['route_sha256'] = None if args.demo else sha256(args.route)
    report['road_source'] = 'synthetic-road-3.5m-wide' if args.demo else str(args.map)
    report['road_sha256'] = None if args.demo else sha256(args.map)
    report['planned_lanelet_route_ids'] = None if args.demo else list(PLANNED_ROUTE_IDS)
    report['validation_road_lanelet_ids'] = None if args.demo else list(VALIDATION_ROAD_IDS)
    report['lanelet_count_in_map'] = n_lanelets
    args.out.mkdir(parents=True, exist_ok=True)
    for profile, data in results.items():
        save_csv(args.out / (profile + '_trajectory.csv'), data['rows'])
        save_csv(args.out / (profile + '_dense_samples.csv'), data['dense'])
    maybe_plot(args.out / 'comparison.png', route, obstacle,
               results['safety_first']['rows'], results['corridor_first']['rows'], cfg)
    plot_report(args.out / 'road_footprints.png', road,
                {p: data['dense'] for p, data in results.items()})
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('\n===== MILESTONE 4E-A: CONSTRAINT-AWARE ROAD GRAPH =====')
    print('Road source:', report['road_source'])
    print('Validation lanelets:', report['validation_road_lanelet_ids'] or 'synthetic region')
    print('Graph states:', report['graph']['time_layered_states'])
    print('Graph edges:', report['graph']['edges'])
    print('Rejected candidate edge attempts (road footprint):',
          report['graph']['edge_rejection_attempt_counts'].get('road_footprint', 0))
    print('Rejected candidate edge attempts (quantization):',
          report['graph']['edge_rejection_attempt_counts'].get('quantization', 0))
    for p, data in results.items():
        cost = data['costs']
        print('{}: R9={:.4f}; R5={:.4f}; R4={:.3f}; final_s={:.1f}m; sampled road={}; inferred steering={:.2f}deg'.format(
            p, cost[0], cost[1], cost[2], data['rows'][-1]['s_m'],
            'PASS' if data['diagnostic']['road_pass'] else 'FAIL',
            data['diagnostic']['max_required_steering_deg']))
    div = report['divergence']
    print('Max time-aligned separation: {:.3f} m'.format(div['max_aligned_separation_m']))
    print('Outputs:', args.out)
    print('LIMIT: sampled footprint screening, NOT physically certified or CARLA executable.')


if __name__ == '__main__':
    main()
