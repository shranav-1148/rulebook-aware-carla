"""Milestone 4D: independent, sampled geometry / steering-demand diagnostics.

AI-authored project-specific analysis (not a CARLA controller or safety certificate).
Reads existing M4B trajectory CSVs and M4A route CSV, plus existing Town01
CommonRoad-compatible XML. Does not change any upstream research code.

The M4B graph's actions are independent longitudinal/lateral accelerations,
not steering commands. We reconstruct the proposed relative motion between
1-second states, estimate yaw from local route tangent + atan2(v_lat, v_long),
and INFER steering demand with a no-slip bicycle-model approximation.
"""
import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from shapely.geometry import Polygon
from shapely.ops import unary_union

from town01_motion_graph import pose_at, read_route_csv, world_xy, synthetic_route

BASE = Path(__file__).resolve().parent
ROUTE_IDS = (39, 38, 40, 58, 60)


@dataclass(frozen=True)
class Limits:
    length_m: float = 4.5
    width_m: float = 1.9
    wheelbase_m: float = 2.7
    max_steering_deg: float = 35.0
    max_lateral_accel_mps2: float = 3.0
    max_forward_accel_mps2: float = 3.0
    max_braking_mps2: float = 6.0
    max_quantization_jump_m: float = 0.05
    max_speed_residual_mps: float = 0.5
    sample_dt_s: float = 0.1
    area_tolerance_m2: float = 1e-5


def verify_limits(limits):
    fields = [limits.length_m, limits.width_m, limits.wheelbase_m,
              limits.max_steering_deg, limits.max_lateral_accel_mps2,
              limits.max_forward_accel_mps2, limits.max_braking_mps2,
              limits.max_quantization_jump_m, limits.max_speed_residual_mps,
              limits.sample_dt_s]
    if any(not math.isfinite(v) or v <= 0 for v in fields):
        raise ValueError('All vehicle limits and sampling settings must be finite and positive')
    if not 0 < limits.max_steering_deg < 90:
        raise ValueError('Steering limit must be between 0 and 90 degrees')
    if limits.area_tolerance_m2 < 0:
        raise ValueError('Area tolerance must be non-negative')


def read_motion_csv(path):
    required = {'t_s', 's_m', 'd_m', 'v_mps', 'lateral_v_mps',
                'x_m', 'y_m', 'ax_mps2', 'ay_mps2'}
    with Path(path).open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f'Missing columns in {path}: {sorted(required - set(reader.fieldnames or []))}')
        data = []
        for raw in reader:
            item = {k: float(raw[k]) for k in ('t_s', 's_m', 'd_m', 'v_mps',
                                               'lateral_v_mps', 'x_m', 'y_m')}
            for k in ('ax_mps2', 'ay_mps2'):
                item[k] = None if raw[k] is None or raw[k].strip() == '' else float(raw[k])
            if not all(math.isfinite(v) for v in item.values() if v is not None):
                raise ValueError(f'Nonfinite value in {path}')
            data.append(item)
    if len(data) < 2 or any(b['t_s'] <= a['t_s'] for a,b in zip(data, data[1:])):
        raise ValueError('Trajectory must contain >=2 strictly increasing timestamps')
    if any(a['ax_mps2'] is None or a['ay_mps2'] is None for a in data[:-1]):
        raise ValueError('Each non-final trajectory state must have an outgoing acceleration action')
    return data


def lanelet_polygon(left_vertices, right_vertices):
    """Polygon from CommonRoad left/right boundaries. Raises on unusable geometry."""
    left = [(float(p[0]), float(p[1])) for p in left_vertices]
    right = [(float(p[0]), float(p[1])) for p in right_vertices]
    if len(left) < 2 or len(right) < 2:
        raise ValueError('Not enough lanelet boundary vertices')
    poly = Polygon(right + list(reversed(left)))
    if poly.is_empty or not poly.is_valid or poly.area <= 1e-9:
        raise ValueError('Invalid lanelet polygon: inspect road geometry; no silent repair')
    return poly


def road_from_lanelets(lanelets, ids=ROUTE_IDS):
    lookup = {int(l.lanelet_id): l for l in lanelets}
    missing = [i for i in ids if i not in lookup]
    if missing:
        raise ValueError(f'Route lanelets absent: {missing}')
    polys = [lanelet_polygon(lookup[i].left_vertices, lookup[i].right_vertices)
             for i in ids]
    road = unary_union(polys)
    if road.is_empty or not road.is_valid:
        raise ValueError('Route road union has invalid geometry')
    return road


def load_town01_road(xml_path, ids=ROUTE_IDS):
    # Lazy import: synthetic tests do not require CommonRoad installed.
    from commonroad.common.file_reader import CommonRoadFileReader
    scenario, _ = CommonRoadFileReader(str(xml_path)).open()
    road = road_from_lanelets(scenario.lanelet_network.lanelets, ids)
    return road, len(scenario.lanelet_network.lanelets)


def vehicle_rectangle(x, y, yaw, length_m, width_m):
    """Assume (x,y) is RECTANGLE CENTER and yaw is aligned with vehicle body."""
    c, s = math.cos(yaw), math.sin(yaw)
    corners = []
    for longitudinal, lateral in ((1,1),(1,-1),(-1,-1),(-1,1)):
        dx = longitudinal * length_m / 2
        dy = lateral * width_m / 2
        corners.append((x + c*dx - s*dy, y + s*dx + c*dy))
    return Polygon(corners)


def inferred_yaw(route, station, speed, lateral_speed):
    """Heading inferred from route tangent and relative velocity, NOT measured yaw.

    Approximation: ignores the (1 - curvature * lateral_offset) factor in the
    exact Frenet-to-world velocity mapping. Appropriate only for gentle curves.
    """
    _, _, route_yaw = pose_at(route, station)
    return route_yaw + math.atan2(lateral_speed, speed)


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def reconstruct_dense(rows, route, sample_dt_s):
    """Resample constant-acceleration M4B edges, excluding snapped endpoints.

    Graph quantizes each next state. The computed end of an edge can differ
    from the snapped state at the start of the next edge; quantify this jump.
    """
    if sample_dt_s <= 0:
        raise ValueError('Invalid sampling interval')
    samples = []
    seams = []
    for a,b in zip(rows, rows[1:]):
        dt = b['t_s']-a['t_s']
        n = max(2, math.ceil(dt / sample_dt_s))
        ax, ay = a['ax_mps2'], a['ay_mps2']
        expected_s = a['s_m'] + a['v_mps']*dt + 0.5*ax*dt*dt
        expected_d = a['d_m'] + a['lateral_v_mps']*dt + 0.5*ay*dt*dt
        jump = math.hypot(expected_s-b['s_m'], expected_d-b['d_m'])
        seams.append({'t_s': b['t_s'], 'position_jump_in_route_frame_m': jump,
                      'speed_jump_mps': math.hypot(a['v_mps']+ax*dt-b['v_mps'],
                                                  a['lateral_v_mps']+ay*dt-b['lateral_v_mps'])})
        for j in range(n):
            tau = dt*j/n
            station = a['s_m'] + a['v_mps']*tau + 0.5*ax*tau*tau
            lateral = a['d_m'] + a['lateral_v_mps']*tau + 0.5*ay*tau*tau
            v = a['v_mps'] + ax*tau
            vy = a['lateral_v_mps'] + ay*tau
            x,y = world_xy(route, station, lateral)
            yaw = inferred_yaw(route, station, v, vy)
            samples.append({'t_s': a['t_s']+tau, 'x_m':x, 'y_m':y,
                            'yaw_rad':yaw, 'v_approx_mps':math.hypot(v,vy),
                            's_m':station, 'd_m':lateral})
    last = rows[-1]
    x,y = world_xy(route, last['s_m'],last['d_m'])
    samples.append({'t_s':last['t_s'], 'x_m':x, 'y_m':y,
                    'yaw_rad':inferred_yaw(route,last['s_m'],last['v_mps'],last['lateral_v_mps']),
                    'v_approx_mps':math.hypot(last['v_mps'],last['lateral_v_mps']),
                    's_m':last['s_m'], 'd_m':last['d_m']})
    return samples, seams


def check_road(samples, road, limits):
    outside = []
    boundary_clearances = []
    for p in samples:
        footprint = vehicle_rectangle(p['x_m'],p['y_m'],p['yaw_rad'],limits.length_m,limits.width_m)
        out_area = footprint.difference(road).area
        outside.append(out_area)
        boundary_clearances.append(footprint.distance(road.boundary) if out_area <= limits.area_tolerance_m2 else 0.0)
        p['outside_area_m2'] = out_area
        p['road_contained'] = (out_area <= limits.area_tolerance_m2)
    count = sum(a > limits.area_tolerance_m2 for a in outside)
    return {'road_pass': count == 0, 'road_failure_samples': count,
            'road_sample_count': len(samples),
            'max_footprint_outside_area_m2': max(outside),
            'min_boundary_clearance_m': min(boundary_clearances)}


def check_dynamics(samples, seams, limits):
    """Inverse no-slip bicycle steering/acceleration DEMAND; not validation of yaw.

    delta_req = atan(L * yaw_rate / speed), and a_lat ~= speed * yaw_rate.
    Yaw is estimated from route-relative velocity. A 1-s discrete graph with
    snapped states can violate instantaneous kinematics; report these errors.
    """
    steering = []
    lateral_accel = []
    pos_accel = []
    neg_accel = []
    speed_residual = []
    for a,b in zip(samples,samples[1:]):
        dt = b['t_s']-a['t_s']
        if dt <= 0:
            raise ValueError('Nonincreasing resampled time')
        avg_v = (a['v_approx_mps']+b['v_approx_mps'])/2
        yawrate = wrap_angle(b['yaw_rad']-a['yaw_rad'])/dt
        required_steer = math.degrees(math.atan2(limits.wheelbase_m*yawrate, max(avg_v,1e-6)))
        steering.append(abs(required_steer))
        lateral_accel.append(abs(avg_v*yawrate))
        accel = (b['v_approx_mps']-a['v_approx_mps'])/dt
        pos_accel.append(max(0,accel))
        neg_accel.append(max(0,-accel))
        dx = b['x_m']-a['x_m']
        dy = b['y_m']-a['y_m']
        speed_residual.append(abs(math.hypot(dx,dy)/dt - avg_v))
    output = {'max_required_steering_deg':max(steering,default=0),
              'max_inferred_lateral_accel_mps2':max(lateral_accel,default=0),
              'max_inferred_forward_accel_mps2':max(pos_accel,default=0),
              'max_inferred_braking_mps2':max(neg_accel,default=0),
              'max_position_speed_residual_mps':max(speed_residual,default=0),
              'max_quantization_position_jump_m':max((z['position_jump_in_route_frame_m'] for z in seams),default=0),
              'max_quantization_speed_jump_mps':max((z['speed_jump_mps'] for z in seams),default=0)}
    output['bicycle_demand_pass'] = (output['max_required_steering_deg'] <= limits.max_steering_deg + 1e-6 and
         output['max_inferred_lateral_accel_mps2'] <= limits.max_lateral_accel_mps2 + 1e-6 and
         output['max_inferred_forward_accel_mps2'] <= limits.max_forward_accel_mps2 + 1e-6 and
         output['max_inferred_braking_mps2'] <= limits.max_braking_mps2 + 1e-6)
    output['quantization_pass'] = (output['max_quantization_position_jump_m'] <= limits.max_quantization_jump_m + 1e-6 and
                                  output['max_quantization_speed_jump_mps'] <= 1e-6)
    output['velocity_position_pass'] = output['max_position_speed_residual_mps'] <= limits.max_speed_residual_mps + 1e-6
    return output


def validate_trajectory(rows, route, road, limits):
    dense, seams = reconstruct_dense(rows,route,limits.sample_dt_s)
    geometry = check_road(dense,road,limits)
    dynamics = check_dynamics(dense,seams,limits)
    flags = [geometry['road_pass'],dynamics['bicycle_demand_pass'],
             dynamics['quantization_pass'],dynamics['velocity_position_pass']]
    result = {**geometry,**dynamics,'all_diagnostic_thresholds_pass':all(flags),
              'dense_samples':len(dense),'original_states':len(rows)}
    return result,dense,seams


def save_csv(path, rows):
    if not rows:
        raise ValueError('Cannot save empty dataset')
    with Path(path).open('w',newline='',encoding='utf-8') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def _plot_polygon(ax,poly,**kwargs):
    if poly.geom_type == 'Polygon':
        x,y=poly.exterior.xy
        ax.fill(x,y,**kwargs)
    elif poly.geom_type == 'MultiPolygon':
        for piece in poly.geoms:
            _plot_polygon(ax,piece,**kwargs)


def plot_report(path,road,series):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig,ax=plt.subplots(figsize=(11,5))
    _plot_polygon(ax,road,color='gray',alpha=0.22,label='Route-lanelet road union')
    for name, samples in series.items():
        ax.plot([s['x_m'] for s in samples],[s['y_m'] for s in samples],label=name,lw=1.8)
        invalid=[s for s in samples if not s['road_contained']]
        if invalid:
            ax.scatter([s['x_m'] for s in invalid],[s['y_m'] for s in invalid],s=17,marker='x',label=f'{name}: off-road footprint')
    ax.set_aspect('equal',adjustable='datalim'); ax.grid(alpha=0.25)
    ax.set_title('Milestone 4D: road polygons + graph-selected motions (diagnostics only)')
    ax.set_xlabel('CommonRoad X (m)');ax.set_ylabel('CommonRoad Y (m)')
    ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)
    return True


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--map',type=Path,default=BASE/'maps/Town01.rpp_compat.xml')
    ap.add_argument('--route',type=Path,default=BASE/'output/town01_graph/route_waypoints.csv')
    ap.add_argument('--safety',type=Path,default=BASE/'output/town01_motion_graph/safety_first_trajectory.csv')
    ap.add_argument('--corridor',type=Path,default=BASE/'output/town01_motion_graph/corridor_first_trajectory.csv')
    ap.add_argument('--out',type=Path,default=BASE/'output/town01_feasibility')
    ap.add_argument('--sample-dt',type=float,default=0.1)
    ap.add_argument('--vehicle-length',type=float,default=4.5)
    ap.add_argument('--vehicle-width',type=float,default=1.9)
    ap.add_argument('--wheelbase',type=float,default=2.7)
    ap.add_argument('--max-steer-deg',type=float,default=35.0)
    ap.add_argument('--max-lateral-accel',type=float,default=3.0)
    ap.add_argument('--demo',action='store_true',help='Synthetic reference road only; not a Town01 geometry test')
    args=ap.parse_args()
    limits=Limits(length_m=args.vehicle_length,width_m=args.vehicle_width,wheelbase_m=args.wheelbase,
                  max_steering_deg=args.max_steer_deg,max_lateral_accel_mps2=args.max_lateral_accel,
                  sample_dt_s=args.sample_dt)
    verify_limits(limits)
    if args.demo:
        route=synthetic_route()
        road=Polygon([(-4,-1.75),(84,-1.75),(84,1.75),(-4,1.75)])
        lanelet_count=None
    else:
        if not args.map.is_file():
            ap.error(f'Town01 map missing: {args.map}')
        route=read_route_csv(args.route)
        road,lanelet_count=load_town01_road(args.map)
    profiles={'R9_first':args.safety,'R5_first':args.corridor}
    findings={}; samples_by_profile={}; seams_by_profile={}
    for name,filename in profiles.items():
        if not filename.is_file():
            ap.error(f'Missing trajectory: {filename}. Run town01_motion_graph.py first.')
        rows=read_motion_csv(filename)
        findings[name],samples_by_profile[name],seams_by_profile[name]=validate_trajectory(rows,route,road,limits)
    args.out.mkdir(parents=True,exist_ok=True)
    for name in profiles:
        save_csv(args.out/f'{name.lower()}_dense_samples.csv',samples_by_profile[name])
        save_csv(args.out/f'{name.lower()}_quantization_seams.csv',seams_by_profile[name])
    plot_saved=plot_report(args.out/'feasibility_overview.png',road,samples_by_profile)
    report={'experiment':'M4D sampled footprint and inverse bicycle-demand diagnostic',
            'town01_geometry_tested':not args.demo,
            'road_source':str(args.map) if not args.demo else 'illustrative synthetic 3.5-m wide road',
            'road_sha256':_sha(args.map) if not args.demo else None,
            'route_source':str(args.route) if not args.demo else 'synthetic 80-m route',
            'route_sha256':_sha(args.route) if not args.demo else None,
            'source_trajectories':{k:{'path':str(v),'sha256':_sha(v)} for k,v in profiles.items()},
            'lanelets_checked':list(ROUTE_IDS) if not args.demo else None,
            'full_map_lanelets':lanelet_count,
            'limits':limits.__dict__, 'results':findings, 'plot_written':plot_saved,
            'limitations':[
                'Not certified vehicle feasibility; yaw is inferred, not simulated.',
                'Road is the union of ONLY the four selected lanelets; adjacent legal lanes not included.',
                'Footprint uses assumed centered rectangle, not measured CARLA actor bounding-box geometry.',
                'Approximate local Frenet yaw ignores curvature-times-offset correction.',
                'Steering and lateral acceleration inferred by discrete yaw differences; sensitive to waypoint spacing and sampling.',
                'No exact bicycle dynamics integration, steering-rate limit, tire friction, obstacle footprint collision, or controller tracking.',
                'Existing M4B graph rounds motion-state variables, which can create discontinuities.',
                'Input XML compatibility copy changed curb markings to unknown: line semantics are not validated.',
                'Passing these diagnostic thresholds does not mean a path is CARLA-executable or safe.'
            ]}
    (args.out/'feasibility_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('\n===== MILESTONE 4D: MOTION FEASIBILITY DIAGNOSTICS =====')
    print('Road:',report['road_source'])
    print('Vehicle rectangle: {:.2f} x {:.2f} m; wheelbase: {:.2f} m'.format(limits.length_m,limits.width_m,limits.wheelbase_m))
    for name,r in findings.items():
        print(f'{name}: road={"PASS" if r["road_pass"] else "FAIL"} ({r["road_failure_samples"]}/{r["road_sample_count"]} footprint samples outside); '
              f'bicycle-demand={"PASS" if r["bicycle_demand_pass"] else "FAIL"}; '
              f'quantization={"PASS" if r["quantization_pass"] else "FAIL"}; '
              f'position-speed={"PASS" if r["velocity_position_pass"] else "FAIL"}')
        print('  max outside area={:.4f} m²; steer={:.2f}°; lateral a={:.2f} m/s²; '
              'seam jump={:.3f} m'.format(r['max_footprint_outside_area_m2'],r['max_required_steering_deg'],
                                         r['max_inferred_lateral_accel_mps2'],r['max_quantization_position_jump_m']))
    print('Outputs:',args.out)
    print('IMPORTANT: screening only; no closed-loop CARLA execution or certified physical feasibility.')


if __name__=='__main__':
    main()
