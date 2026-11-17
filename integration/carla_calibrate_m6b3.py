"""Milestone 6B-3: guarded, low-speed CARLA dynamics calibration (NO replanning).

Default offline preview. --inspect-server reads only. --execute-straight / --execute-steering
are EXPLICIT world-mutating, tick-owning driving modes requiring --confirm-drive.
Only our spawned Tesla Model 3 is controlled or destroyed. No cone is spawned.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
DEFAULT_OBSERVATION = HERE / 'output/carla_observation_6b2_stable/trial_003/live_observation.json'
DEFAULT_OUT = HERE / 'output/carla_calibration_6b3'
EXPECTED_START_CR = (384.591, 1.99335)
ASSUMED_WHEELBASE_M = 2.7
DT = 0.05
MAX_SIM_DRIVE_S = 4.2
MAX_SPEED_MPS = 3.5
MAX_DISTANCE_M = 8.0
MAX_LATERAL_M = 0.65
MAX_HEADING_DEG = 12.0
# The throttle pulse must demonstrate measurable motion before declaring a usable calibration.
MIN_THROTTLE_PEAK_FORWARD_MPS = 0.05
MIN_THROTTLE_DISPLACEMENT_M = 0.02
# Require a genuine *measured* yaw/wheel-angle response to a short steering pulse.
MIN_STEERING_PULSE_SPEED_MPS = 0.10
MIN_STEERING_PULSE_HEADING_DEG = 0.10
MIN_STEERING_PULSE_WHEEL_DEG = 0.25
# Conservative XY envelope for the short 8 m forward calibration, including the
# ~4.8 m vehicle footprint and heading/lateral drift. This is a *screen*, not
# a collision-free-space certification.
PROP_STATION_MIN_M = -4.0
PROP_STATION_MAX_M = 12.0
PROP_LATERAL_HALF_WIDTH_M = 4.0


def finite(name, x):
    try:
        v = float(x)
    except (TypeError, ValueError) as exc:
        raise ValueError(name + ' is not numeric') from exc
    if not math.isfinite(v):
        raise ValueError(name + ' is not finite')
    return v


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def read_stable_observation(path):
    """Use prior stable capture as nominal *placement*, not as a live actor handle."""
    path = Path(path)
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('stability_capture_mode') != 'passive_multi_frame':
        raise ValueError('Requires 6B-2S multi-frame stable capture')
    if record.get('cleanup_ok') is not True:
        raise ValueError('Previous capture actor cleanup was not confirmed')
    if (record.get('stability') or {}).get('latest', {}).get('accepted') is not True:
        raise ValueError('Previous capture did not pass physical stability gate')
    checks = record.get('checks') or {}
    required = ('ego_pose_matches_nominal_start', 'obstacle_center_matches_offline_location',
                'ego_within_route_offset_bound', 'ego_within_route_heading_bound',
                'initial_wheel_steering_near_zero', 'small_lateral_velocity')
    if not all(checks.get(k) is True for k in required):
        raise ValueError('Previous stable observation failed scenario consistency checks')
    server = record.get('server') or {}
    if server.get('map') != 'Carla/Maps/Town01':
        raise ValueError('Previous observation was not CARLA Town01')
    if server.get('carla_client_version') != '0.9.16' or server.get('carla_server_version') != '0.9.16':
        raise ValueError('Previous observation CARLA version mismatch')
    ego = record['ego']
    xy = ego['body_center_cr_xy_m']
    if len(xy) != 2:
        raise ValueError('Invalid initial body-center array')
    x, y = finite('body_x', xy[0]), finite('body_y', xy[1])
    yaw = finite('yaw_cr_rad', ego['yaw_cr_rad'])
    vx = finite('forward_speed', ego['longitudinal_speed_mps'])
    vy = finite('lateral_speed', ego['lateral_speed_mps'])
    vz = finite('vertical_speed', ego['velocity_carla_xyz_mps'][2])
    if (math.hypot(x-EXPECTED_START_CR[0], y-EXPECTED_START_CR[1]) > 0.25
            or abs(math.degrees(wrap(yaw-math.pi))) > 3.0):
        raise ValueError('Recorded state is not near the validated westbound Town01 start')
    if max(abs(vx),abs(vy),abs(vz)) > 0.05 + 1e-8:
        raise ValueError('Recorded observation is not approximately at rest')
    return {'body_center_cr_xy_m': [x, y], 'yaw_cr_rad': yaw,
            'source_frame': int(record['snapshot']['frame']),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'path': str(path.resolve())}


def actor_origin_for_center(cx, cy, yaw, box_local):
    bx, by = finite('bbox offset x', box_local.x), finite('bbox offset y',box_local.y)
    return (cx - bx*math.cos(yaw)+by*math.sin(yaw),
            cy - bx*math.sin(yaw)-by*math.cos(yaw))


def center_from_actor(x, y, yaw, box_local):
    return (x + box_local.x*math.cos(yaw)-box_local.y*math.sin(yaw),
            y + box_local.x*math.sin(yaw)+box_local.y*math.cos(yaw))


def sample_actor(world, actor, carla, box_local, expected_frame=None):
    """Pose and velocity from SAME WorldSnapshot. Wheel physics angles separately sampled."""
    snap = world.get_snapshot()
    if expected_frame is not None and int(snap.frame) != int(expected_frame):
        raise RuntimeError('World snapshot frame differs from our explicit tick; another client may be ticking')
    a = snap.find(actor.id)
    if a is None:
        raise RuntimeError('Owned ego missing from snapshot frame %s' % snap.frame)
    pose = a.get_transform()
    v = a.get_velocity()
    accel = a.get_acceleration()
    yaw_ca = math.radians(finite('carla_yaw',pose.rotation.yaw))
    yaw_cr = wrap(-yaw_ca)
    cx,cy = center_from_actor(finite('actor x',pose.location.x), finite('actor y',pose.location.y), yaw_ca, box_local)
    vx,vy,vz = finite('velocity x',v.x),finite('velocity y',v.y),finite('velocity z',v.z)
    fwd = vx*math.cos(yaw_ca)+vy*math.sin(yaw_ca)
    lat = -vx*math.sin(yaw_ca)+vy*math.cos(yaw_ca)
    wl = finite('FL steer', actor.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel))
    wr = finite('FR steer', actor.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel))
    return {'frame':int(snap.frame),'t_sim_s':float(snap.timestamp.elapsed_seconds),
            'x_cr_m':cx,'y_cr_m':-cy,'yaw_cr_rad':yaw_cr,
            'forward_speed_mps':fwd,'lateral_speed_mps':lat,'vertical_speed_mps':vz,
            'total_speed_mps':math.sqrt(vx*vx+vy*vy+vz*vz),
            'acceleration_xyz_carla_mps2':[float(accel.x),float(accel.y),float(accel.z)],
            'wheel_front_left_deg':wl,'wheel_front_right_deg':wr,
            'wheel_angle_timestamp_warning':'Wheel physics is queried from actor, not frame-stamped snapshot'}


def plan_command(mode, t, steer_sign=1):
    """Fixed low-speed, open-loop commands. Independent of planned Rulebook paths."""
    if mode in ('straight', 'manual_first'):
        if t < 1.5:
            return ('throttle', 0.25, 0.0, 0.0)
        if t < 2.0:
            return ('coast', 0.0, 0.0, 0.0)
        return ('brake', 0.0, 0.0, 0.65)
    if mode in ('steering','manual_steering'):
        if t < 1.1:
            return ('throttle_straight',0.25,0.,0.)
        if t < 1.8:
            return ('steer_pulse',0.20,0.08*steer_sign,0.)
        if t < 2.2:
            return ('neutralize',0.0,0.,0.)
        return ('brake',0.,0.,0.65)
    raise ValueError('Unknown calibration mode')


def commanded_vehicle_control(carla, mode, throttle, steer, brake):
    """Explicit, temporary manual-first diagnostic. No vehicle physics mutation.

    For normal 'straight' / 'steering', preserve the previous automatic mode.
    """
    if mode not in ('straight', 'steering', 'manual_first', 'manual_steering'):
        raise ValueError('Unsupported control mode')
    manual = mode in ('manual_first', 'manual_steering')
    return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake,
                                hand_brake=False, reverse=False,
                                manual_gear_shift=manual, gear=1 if manual else 0)


def safe_failure_state(actor):
    """Read failure diagnostics without making it a requirement on older mock APIs."""
    try:
        return str(actor.get_failure_state())
    except (AttributeError, RuntimeError, TypeError):
        return 'unavailable'


def estimated_wheel_geometry(physics):
    """Estimate axle-to-axle length using CARLA world-space wheel positions in cm.

    This is geometric and not an independently verified effective kinematic wheelbase.
    """
    try:
        w = physics.wheels
        if len(w) != 4:
            return {'available':False,'reason':'wheel count is not 4'}
        pts = [(finite('wheel.x',p.position.x)/100.0,
                finite('wheel.y',p.position.y)/100.0) for p in w]
        front = ((pts[0][0]+pts[1][0])/2, (pts[0][1]+pts[1][1])/2)
        rear = ((pts[2][0]+pts[3][0])/2, (pts[2][1]+pts[3][1])/2)
        return {'available':True,'wheelbase_from_xy_m':math.hypot(front[0]-rear[0],front[1]-rear[1]),
                'front_center_carla_xy_m':list(front),'rear_center_carla_xy_m':list(rear),
                'source':'get_physics_control().wheels[].position divided by 100 (world cm to m)'}
    except (AttributeError, TypeError, ValueError) as e:
        return {'available':False,'reason':str(e)}


def applied_control_readback(actor):
    """Read back the last applied CARLA vehicle control (not just our requested command).

    CARLA get_control() describes the last control processed by the server. It is
    collected after an explicit tick, but is not part of the actor pose snapshot.
    """
    c = actor.get_control()
    return {'applied_throttle': finite('applied throttle', c.throttle),
            'applied_steer': finite('applied steer', c.steer),
            'applied_brake': finite('applied brake', c.brake),
            'applied_hand_brake': bool(c.hand_brake),
            'applied_reverse': bool(c.reverse),
            'applied_manual_gear_shift': bool(c.manual_gear_shift),
            'applied_gear': int(c.gear)}


def motion_during_throttle(rows):
    """Diagnostic gate: a 1.5s positive throttle pulse must produce measured motion."""
    pulse = [r for r in rows if r.get('phase') in ('throttle','throttle_straight')]
    if not pulse:
        return {'passed': False, 'reason': 'No straight throttle-phase samples'}
    peak = max(r['forward_speed_mps'] for r in pulse)
    distance = max(r['distance_from_start_m'] for r in pulse)
    good = peak >= MIN_THROTTLE_PEAK_FORWARD_MPS and distance >= MIN_THROTTLE_DISPLACEMENT_M
    return {'passed':good, 'peak_forward_speed_mps':peak,
            'max_displacement_during_throttle_m':distance,
            'required_peak_forward_speed_mps': MIN_THROTTLE_PEAK_FORWARD_MPS,
            'required_displacement_m':MIN_THROTTLE_DISPLACEMENT_M}


def steering_pulse_gate(rows):
    """Reject 'completed' when steering was commanded but no turn was measured.

    Data remain diagnostics: wheel angles are read from the actor separately
    from the frame-stamped vehicle pose/velocity snapshot.
    """
    pulse = [r for r in rows if r.get('phase') == 'steer_pulse']
    if not pulse:
        return {'passed':False,'reason':'No steering-pulse samples'}
    peak_speed = max(r['forward_speed_mps'] for r in pulse)
    peak_wheel = max(abs(r['wheel_front_left_deg']) for r in pulse)
    peak_wheel = max(peak_wheel, max(abs(r['wheel_front_right_deg']) for r in pulse))
    peak_yaw = max(abs(r['heading_from_start_deg']) for r in pulse)
    peak_cmd = max(abs(r['applied_steer']) for r in pulse)
    good = (peak_speed >= MIN_STEERING_PULSE_SPEED_MPS and
            peak_wheel >= MIN_STEERING_PULSE_WHEEL_DEG and
            peak_yaw >= MIN_STEERING_PULSE_HEADING_DEG and
            peak_cmd >= 0.06)
    return {'passed':good,'peak_forward_speed_during_steering_mps':peak_speed,
            'peak_abs_front_wheel_deg':peak_wheel,
            'peak_abs_heading_change_deg':peak_yaw,
            'peak_abs_applied_steer':peak_cmd,
            'minimum_speed_mps':MIN_STEERING_PULSE_SPEED_MPS,
            'minimum_wheel_deg':MIN_STEERING_PULSE_WHEEL_DEG,
            'minimum_heading_change_deg':MIN_STEERING_PULSE_HEADING_DEG}


def manual_steering_control_check(row, phase, throttle, steer, brake):
    """Verify applied-command readback on every manual steering control tick."""
    if (abs(row['applied_throttle']-throttle) > .02 or
        abs(row['applied_steer']-steer) > .02 or
        abs(row['applied_brake']-brake) > .05 or
        row['applied_hand_brake'] or row['applied_reverse'] or
        not row['applied_manual_gear_shift'] or row['applied_gear'] != 1):
        raise RuntimeError('Manual steering applied-control mismatch in phase '+phase)


def evaluate_limits(row):
    reasons=[]
    if row['total_speed_mps'] > MAX_SPEED_MPS or row['forward_speed_mps'] < -0.20:
        reasons.append('speed/reverse bound')
    if row['distance_from_start_m'] > MAX_DISTANCE_M:
        reasons.append('displacement bound')
    if abs(row['cross_track_m']) > MAX_LATERAL_M:
        reasons.append('lateral corridor bound')
    if abs(row['heading_from_start_deg']) > MAX_HEADING_DEG:
        reasons.append('heading change bound')
    if abs(row['vertical_speed_mps']) > 0.5:
        reasons.append('unexpected vertical motion')
    return reasons


def enrich(sample, start, command, t):
    x0,y0,yaw0=start['x_cr_m'],start['y_cr_m'],start['yaw_cr_rad']
    dx,dy=sample['x_cr_m']-x0,sample['y_cr_m']-y0
    phase,throttle,steer,brake=command
    return dict(sample,t_s=round(t,5),phase=phase,command_throttle=throttle,
                command_steer_normalized=steer,command_brake=brake,
                distance_from_start_m=math.hypot(dx,dy),
                along_track_m=dx*math.cos(yaw0)+dy*math.sin(yaw0),
                cross_track_m=-dx*math.sin(yaw0)+dy*math.cos(yaw0),
                heading_from_start_deg=math.degrees(wrap(sample['yaw_cr_rad']-yaw0)))


def kinematic_model_comparison(rows, wheelbase=ASSUMED_WHEELBASE_M):
    """Two sign hypotheses, using MEASURED speed as exogenous input.

    This tests steering/yaw geometry only; it is NOT an acceleration prediction.
    """
    if len(rows)<2:
        return {'samples':len(rows),'sign_hypothesis_rmse_m':{}}
    sums={'+':[], '-':[]}; yaw_err={'+':[],'-':[]}
    for sign,label in ((1,'+'),(-1,'-')):
        x,y=rows[0]['x_cr_m'],rows[0]['y_cr_m']
        yaw=rows[0]['yaw_cr_rad']
        for prev,cur in zip(rows,rows[1:]):
            dt=cur['t_s']-prev['t_s']
            if dt<=0:
                raise ValueError('Nonmonotonic calibration times')
            speed=max(0.,0.5*(prev['forward_speed_mps']+cur['forward_speed_mps']))
            delta=math.radians(abs(0.5*(cur['wheel_front_left_deg']+cur['wheel_front_right_deg'])))
            yaw_next=wrap(yaw+sign*dt*speed*math.tan(delta)/wheelbase)
            yaw_mid=wrap(yaw+0.5*wrap(yaw_next-yaw))
            x+=dt*speed*math.cos(yaw_mid)
            y+=dt*speed*math.sin(yaw_mid)
            yaw=yaw_next
            sums[label].append(math.hypot(x-cur['x_cr_m'],y-cur['y_cr_m']))
            yaw_err[label].append(abs(math.degrees(wrap(yaw-cur['yaw_cr_rad']))))
    return {'speed_input':'measured forward velocity (not predicted engine acceleration)',
            'wheelbase_assumed_m':wheelbase,
            'steer_magnitude_source':'mean absolute signed front-wheel physics angle; sign tested both ways',
            'steering_angles_not_frame_locked':True,
            'sign_hypothesis_rmse_m':{k: math.sqrt(mean(z*z for z in vals)) for k,vals in sums.items()},
            'sign_hypothesis_heading_rmse_deg':{k: math.sqrt(mean(z*z for z in vals)) for k,vals in yaw_err.items()},
            'note':'Two sign hypotheses are diagnostics; a single short run does not establish a calibrated wheelbase.'}


def summarize(rows, mode, status, abort_reason, collisions):
    out={'status':status,'abort_reason':abort_reason,'mode':mode,'samples':len(rows),
         'collision_events':list(collisions), 'collision_count':len(collisions),
         'wheelbase_assumed_m':ASSUMED_WHEELBASE_M,
         'carla_control_applied':bool(rows and len(rows)>1),
         'rulebook_replanning_enabled':False,'obstacle_spawned':False,
         'limits':{'max_speed_mps':MAX_SPEED_MPS,'max_displacement_m':MAX_DISTANCE_M,
                   'max_cross_track_m':MAX_LATERAL_M,'max_heading_change_deg':MAX_HEADING_DEG}}
    if rows:
        out['throttle_motion_gate'] = motion_during_throttle(rows) if mode in ('straight','manual_first','manual_steering') else None
        out['steering_pulse_gate'] = steering_pulse_gate(rows) if mode == 'manual_steering' else None
        out.update({'duration_s':rows[-1]['t_s'],
                    'final_forward_speed_mps':rows[-1]['forward_speed_mps'],
                    'max_total_speed_mps':max(row['total_speed_mps'] for row in rows),
                    'max_displacement_m':max(row['distance_from_start_m'] for row in rows),
                    'max_abs_cross_track_m':max(abs(row['cross_track_m']) for row in rows),
                    'max_abs_heading_change_deg':max(abs(row['heading_from_start_deg']) for row in rows),
                    'kinematic_comparison':kinematic_model_comparison(rows)})
        if len(rows)>1:
            out['measured_speed_response']={'peak_forward_speed_mps':max(r['forward_speed_mps'] for r in rows),
                'forward_accel_during_throttle_mps2':(next((r['forward_speed_mps'] for r in reversed(rows) if r['phase'].startswith('throttle')),rows[0]['forward_speed_mps'])-rows[0]['forward_speed_mps'])/
                max(0.05,next((r['t_s'] for r in reversed(rows) if r['phase'].startswith('throttle')),0.05))}
    return out


def validate_baseline(path, observation_hash, *, expected_mode='straight'):
    doc=json.loads(Path(path).read_text(encoding='utf-8'))
    if (doc.get('mode') != expected_mode or doc.get('status') != 'completed' or
        doc.get('cleanup_ok') is not True or doc.get('settings_restored') is not True or
        doc.get('collision_count') != 0 or doc.get('observation_sha256') != observation_hash or
        finite('baseline displacement',doc.get('max_displacement_m')) < 0.5 or
        doc.get('throttle_motion_gate',{}).get('passed') is not True or
        finite('baseline max speed',doc.get('max_total_speed_mps')) > MAX_SPEED_MPS):
        raise ValueError('Steering mode requires a successful, matching, low-speed straight baseline report')
    return doc


def save_results(out, rows, summary):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    report=out/'summary.json'; samples=out/'calibration_samples.csv'
    if report.exists() or samples.exists():
        raise FileExistsError('Refusing to overwrite existing results at '+str(out))
    if rows:
        with samples.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
            writer.writeheader();writer.writerows(rows)
    report.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report



def inspect_static_props(props, start_x, start_y, yaw_carla_rad):
    """Identify corridor-overlapping props using each actor's world bounding box.

    Only distant `static.prop.mesh` may remain in a driving session. Unknown
    geometry and any other static.prop types remain blockers. XY test deliberately
    includes a generous margin around the expected swept calibration footprint.
    """
    ux, uy = math.cos(yaw_carla_rad), math.sin(yaw_carla_rad)
    nearby, other_types, uninspectable = [], [], []
    permitted_far_mesh = 0
    for actor in props:
        ident = getattr(actor, 'id', None)
        typ = str(getattr(actor, 'type_id', '<unknown>'))
        if typ != 'static.prop.mesh':
            other_types.append({'actor_id': ident, 'type_id': typ})
        try:
            box = actor.bounding_box
            vertices = box.get_world_vertices(actor.get_transform())
            if len(vertices) != 8:
                raise ValueError('expected eight world bounding-box vertices')
            st, lat = [], []
            for v in vertices:
                dx = finite('prop vertex x', v.x) - start_x
                dy = finite('prop vertex y', v.y) - start_y
                st.append(dx * ux + dy * uy)
                lat.append(-dx * uy + dy * ux)
            smin, smax = min(st), max(st)
            lmin, lmax = min(lat), max(lat)
            if (smax >= PROP_STATION_MIN_M and smin <= PROP_STATION_MAX_M and
                    lmax >= -PROP_LATERAL_HALF_WIDTH_M and lmin <= PROP_LATERAL_HALF_WIDTH_M):
                nearby.append({'actor_id': ident, 'type_id': typ,
                               'station_range_m': [smin, smax],
                               'lateral_range_m': [lmin, lmax]})
            elif typ == 'static.prop.mesh':
                permitted_far_mesh += 1
        except Exception as exc:
            uninspectable.append({'actor_id': ident, 'type_id': typ,
                                   'error': type(exc).__name__ + ': ' + str(exc)})
    return {'total': len(props), 'permitted_distant_mesh_count': permitted_far_mesh,
            'corridor': {'station_range_m': [PROP_STATION_MIN_M, PROP_STATION_MAX_M],
                         'lateral_range_m': [-PROP_LATERAL_HALF_WIDTH_M, PROP_LATERAL_HALF_WIDTH_M]},
            'nearby_count': len(nearby), 'nearby_first_12': nearby[:12],
            'unexpected_type_count': len(other_types), 'unexpected_types_first_12': other_types[:12],
            'uninspectable_count': len(uninspectable), 'uninspectable_first_12': uninspectable[:12],
            'clear_for_calibration': not nearby and not other_types and not uninspectable}


def preflight(args, obs, drive=False):
    try:
        import carla
    except ImportError as exc:
        raise RuntimeError('Use Windows .venv containing CARLA 0.9.16 Python API') from exc
    client=carla.Client(args.host,args.port)
    client.set_timeout(args.timeout)
    cv,sv=str(client.get_client_version()),str(client.get_server_version())
    if cv!='0.9.16' or sv!='0.9.16':
        raise RuntimeError('CARLA 0.9.16 client/server required; found '+cv+'/'+sv)
    world=client.get_world()
    wm=world.get_map()
    if wm.name.split('/')[-1]!='Town01':
        raise RuntimeError('Expected CARLA Town01; found '+wm.name)
    bp=world.get_blueprint_library().find('vehicle.tesla.model3')
    if bp.id!='vehicle.tesla.model3':
        raise RuntimeError('Unexpected vehicle blueprint')
    x,y=obs['body_center_cr_xy_m']; yaw=obs['yaw_cr_rad']
    cx,cy=x,-y; yaw_carla=-yaw
    wp=wm.get_waypoint(carla.Location(cx,cy,0),project_to_road=True,lane_type=carla.LaneType.Driving)
    if wp is None:
        raise RuntimeError('No nearby Town01 driving lane')
    wp_t=wp.transform
    err=math.hypot(cx-wp_t.location.x,cy-wp_t.location.y)
    heading=abs(math.degrees(wrap(yaw_carla-math.radians(wp_t.rotation.yaw))))
    if err>0.25 or heading>3.0:
        raise RuntimeError('CARLA waypoint mismatch: lateral %.3fm, yaw %.3fdeg' %(err,heading))
    # Inspect an 8m forward corridor; it should still be the known Town01 lane.
    route_checks=[]
    for s in (2.,4.,6.,8.):
        tx,ty=cx+s*math.cos(yaw_carla),cy+s*math.sin(yaw_carla)
        wp2=wm.get_waypoint(carla.Location(tx,ty,0),project_to_road=True,lane_type=carla.LaneType.Driving)
        if wp2 is None:
            raise RuntimeError('Missing lane waypoint at %.1fm ahead' % s)
        d=math.hypot(wp2.transform.location.x-tx,wp2.transform.location.y-ty)
        h=abs(math.degrees(wrap(yaw_carla-math.radians(wp2.transform.rotation.yaw))))
        if d>0.45 or h>5.0:
            raise RuntimeError('Calibration lane is not straight/aligned at %.1fm: %.3fm, %.2fdeg' %(s,d,h))
        route_checks.append({'station_ahead_m':s,'lane_center_error_m':d,'heading_error_deg':h})
    actors=world.get_actors()
    vehicle_count=len(list(actors.filter('vehicle.*')))
    walker_count=len(list(actors.filter('walker.pedestrian.*')))
    prop_actors=list(actors.filter('static.prop.*'))
    prop_screen=inspect_static_props(prop_actors, cx, cy, yaw_carla)
    spawned_props=len(prop_actors)
    sync=bool(world.get_settings().synchronous_mode)
    info={'map':wm.name,'client_version':cv,'server_version':sv,
          'existing_vehicles':vehicle_count,'existing_walkers':walker_count,
          'existing_props':spawned_props,'world_already_synchronous':sync,
          'static_prop_screen':prop_screen,
          'start_lane_center_distance_m':err,'start_heading_error_deg':heading,
          'ahead_checks':route_checks,'actions_performed':'read_only'}
    if drive and (vehicle_count or walker_count or sync or not prop_screen['clear_for_calibration']):
        raise RuntimeError('Refusing drive: vehicles=%s walkers=%s props=%s (nearby=%s unexpected_types=%s uninspectable=%s) sync=%s. Run --inspect-server for details.' %
                           (vehicle_count, walker_count, spawned_props,
                            prop_screen['nearby_count'], prop_screen['unexpected_type_count'],
                            prop_screen['uninspectable_count'], sync))
    return carla, client, world, bp, wp_t, info


def _get_tick(world, *, timeout_s=10.):
    frame = world.tick(timeout_s)
    return int(frame)


def drive(args, obs, mode):
    carla,client,world,bp,wp_t,pre=preflight(args,obs,drive=True)
    screen=pre['static_prop_screen']
    print('MILESTONE 6B-3 PROP SCREEN: total=%s, allowed_distant_mesh=%s, nearby=%s, unexpected=%s, uninspectable=%s' %
          (screen['total'], screen['permitted_distant_mesh_count'], screen['nearby_count'],
           screen['unexpected_type_count'], screen['uninspectable_count']))
    original_settings=world.get_settings()
    settings_modified=False
    ego=sensor=None
    collision_events=[]; rows=[]
    status='aborted'; abort_reason=None
    warmup_control_count=0; drive_control_count=0; brake_cleanup_attempted=False
    warmup_control_readback=None
    cleanup={'vehicle_destroy_returned_success':False,'sensor_destroy_returned_success':False}
    failure_state_at_cleanup='unavailable'
    config={'fixed_dt_s':DT,'mode':mode,'max_sim_drive_s':MAX_SIM_DRIVE_S,
            'no_velocity_seeding':True,'no_autopilot':True,'no_obstacle_actor':True,
            'wheelbase_assumed_m':ASSUMED_WHEELBASE_M,
            'gear_diagnostic':mode in ('manual_first','manual_steering'),
            'gear_diagnostic_note':'Manual first-gear actuation; automatic gearbox remains unvalidated'}
    if mode in ('manual_steering','steering'):
        baseline = getattr(args,'baseline_report',None)
        repeated = getattr(args,'repeat_baseline_report',None)
        config['baseline_report_sha256'] = hashlib.sha256(Path(baseline).read_bytes()).hexdigest() if baseline else None
        config['repeat_baseline_report_sha256'] = hashlib.sha256(Path(repeated).read_bytes()).hexdigest() if repeated else None
        config['steering_sign_command'] = args.steer_sign
    try:
        settings=world.get_settings()
        settings.synchronous_mode=True
        settings.fixed_delta_seconds=DT
        settings_modified=True  # attempt restore even if apply_settings partially fails
        world.apply_settings(settings)
        x,y=obs['body_center_cr_xy_m']; yaw=-obs['yaw_cr_rad']
        bp.set_attribute('role_name','milestone6b3_calibration') if bp.has_attribute('role_name') else None
        provisional=carla.Transform(carla.Location(x,-y,wp_t.location.z+0.5),carla.Rotation(yaw=math.degrees(yaw)))
        ego=world.try_spawn_actor(bp,provisional)
        if ego is None:
            raise RuntimeError('Could not spawn ego at validated location')
        bb=ego.bounding_box.location
        ax,ay=actor_origin_for_center(x,-y,yaw,bb)
        ego.set_transform(carla.Transform(carla.Location(ax,ay,wp_t.location.z+0.5),carla.Rotation(yaw=math.degrees(yaw))))
        ego.set_autopilot(False)
        ego.set_simulate_physics(True)
        sensorbp=world.get_blueprint_library().find('sensor.other.collision')
        sensor=world.spawn_actor(sensorbp,carla.Transform(),attach_to=ego)
        if sensor is None:
            raise RuntimeError('Could not spawn collision sensor')
        sensor.listen(lambda e:collision_events.append({'frame':int(e.frame),'other_actor_id':int(e.other_actor.id),
                                                         'other_actor_type':str(e.other_actor.type_id)}))
        # Warmup 0.6s minimum; require sustained low motion. No starting speed injection.
        settled=0
        warm_start=None
        for i in range(80):
            ego.apply_control(carla.VehicleControl(throttle=0.,steer=0.,brake=0.5))
            warmup_control_count+=1
            f=_get_tick(world,timeout_s=args.timeout)
            a=sample_actor(world,ego,carla,bb,expected_frame=f)
            if warm_start is None:
                warm_start=a['t_sim_s']
            ok=(abs(a['vertical_speed_mps'])<=0.05 and
                math.hypot(a['forward_speed_mps'],a['lateral_speed_mps'])<=0.05)
            settled=settled+1 if ok else 0
            if collision_events:
                raise RuntimeError('Collision during warmup')
            if a['t_sim_s']-warm_start>=0.6 and settled>=10:
                break
        else:
            raise RuntimeError('Vehicle did not settle in 4 simulated seconds')
        warmup_control_readback=applied_control_readback(ego)
        start=sample_actor(world,ego,carla,bb)
        if (math.hypot(start['x_cr_m']-x,start['y_cr_m']-y)>0.20 or
            abs(math.degrees(wrap(start['yaw_cr_rad']-obs['yaw_cr_rad'])))>2.0):
            raise RuntimeError('Post-spawn ego pose mismatch')
        if abs(start['forward_speed_mps'])>0.05 or abs(start['vertical_speed_mps'])>0.05:
            raise RuntimeError('Vehicle is not stationary at start of calibration')
        physics=ego.get_physics_control()
        front_max=[float(w.max_steer_angle) for w in physics.wheels[:2]]
        if min(front_max)<=0:
            raise RuntimeError('Invalid physics front-wheel steering limit')
        config['physics_front_max_steer_angle_deg']=front_max
        config['vehicle_mass_kg']=float(physics.mass)
        config['estimated_wheel_geometry']=estimated_wheel_geometry(physics)
        config['failure_state_at_start']=safe_failure_state(ego)
        zero=enrich(start,start,('initial',0.,0.,0.),0.)
        zero.update(applied_control_readback(ego))
        rows.append(zero)
        steps=int(math.ceil(MAX_SIM_DRIVE_S/DT))
        for j in range(1,steps+1):
            tprev=(j-1)*DT
            command=plan_command(mode,tprev,args.steer_sign)
            phase,th,steer,br=command
            ego.apply_control(commanded_vehicle_control(carla,mode,th,steer,br))
            drive_control_count+=1
            frame=_get_tick(world,timeout_s=args.timeout)
            a=sample_actor(world,ego,carla,bb,expected_frame=frame)
            row=enrich(a,start,command,j*DT)
            row.update(applied_control_readback(ego))
            rows.append(row)
            # A different client or a vehicle/gear override can leave requested
            # throttle in the CSV while the server actually holds brakes.
            if mode == 'manual_steering' and j >= 2:
                manual_steering_control_check(row,phase,th,steer,br)
            if mode != 'manual_steering' and phase.startswith('throttle') and j >= 2 and (
                    row['applied_throttle'] < th - 0.02 or
                    row['applied_brake'] > 0.05 or
                    row['applied_hand_brake'] or row['applied_reverse'] or
                    row['applied_manual_gear_shift'] != (mode == 'manual_first') or
                    (mode == 'manual_first' and row['applied_gear'] != 1)):
                raise RuntimeError('Throttle request does not match applied CARLA control; inspect readback fields')
            if mode == 'manual_steering' and j == round(1.1 / DT):
                measured = motion_during_throttle(rows)
                if not measured['passed']:
                    raise RuntimeError('No measurable manual-first acceleration before steering; aborting')
            if mode == 'manual_steering' and j == round(1.8 / DT):
                steering_gate = steering_pulse_gate(rows)
                if not steering_gate['passed']:
                    raise RuntimeError('No measurable steering response; inspect front-wheel and yaw telemetry')
            if mode in ('straight','manual_first') and j == round(1.5 / DT):
                measured = motion_during_throttle(rows)
                if not measured['passed']:
                    raise RuntimeError('No measurable acceleration from throttle; calibration invalid '
                                       '(peak forward speed %.6g m/s, displacement %.6g m). '
                                       'Inspect applied control and gear fields before changing actuation.' %
                                       (measured['peak_forward_speed_mps'],
                                        measured['max_displacement_during_throttle_m']))
            bad=evaluate_limits(row)
            if collision_events:
                raise RuntimeError('Collision sensor reported %d event(s)' % len(collision_events))
            if bad:
                raise RuntimeError('Calibration abort: '+', '.join(bad))
            if j % 10==0:
                print('t=%.2fs, speed=%.3fm/s, dx=%.3fm, cross=%.3fm, yaw=%.2fdeg, phase=%s' %
                      (row['t_s'],row['forward_speed_mps'],row['along_track_m'],row['cross_track_m'],row['heading_from_start_deg'],phase))
        # Require end-of-brake low motion (not guaranteed by brake input alone).
        if abs(rows[-1]['forward_speed_mps'])>0.15:
            raise RuntimeError('Vehicle did not stop sufficiently after brake phase')
        if mode in ('straight','manual_first','manual_steering') and not motion_during_throttle(rows)['passed']:
            raise RuntimeError('Cannot mark straight calibration completed: no measurable throttle response')
        if mode == 'manual_steering' and not steering_pulse_gate(rows)['passed']:
            raise RuntimeError('Cannot complete steering calibration: no measured turn')
        status='completed'
    except KeyboardInterrupt:
        abort_reason='interrupted by user'
    except Exception as exc:
        abort_reason=type(exc).__name__+': '+str(exc)
    finally:
        if ego is not None:
            failure_state_at_cleanup=safe_failure_state(ego)
            try:
                ego.apply_control(carla.VehicleControl(throttle=0.,steer=0.,brake=1.0))
                brake_cleanup_attempted=True
                if settings_modified:
                    _get_tick(world,timeout_s=args.timeout)  # one explicit brake tick
            except Exception as exc:
                cleanup['brake_tick_error']=str(exc)
        if sensor is not None:
            try:
                sensor.stop()
                cleanup['sensor_destroy_returned_success']=bool(sensor.destroy())
            except Exception as exc:
                cleanup['sensor_cleanup_error']=str(exc)
        else:
            cleanup['sensor_destroy_returned_success']=True
        if ego is not None:
            try:
                cleanup['vehicle_destroy_returned_success']=bool(ego.destroy())
            except Exception as exc:
                cleanup['vehicle_cleanup_error']=str(exc)
        else:
            cleanup['vehicle_destroy_returned_success']=True
        if settings_modified:
            try:
                world.apply_settings(original_settings)
                now=world.get_settings()
                cleanup['settings_restored']=(
                    bool(now.synchronous_mode)==bool(original_settings.synchronous_mode)
                    and now.fixed_delta_seconds==original_settings.fixed_delta_seconds)
            except Exception as exc:
                cleanup['settings_restored']=False
                cleanup['world_restore_error']=str(exc)
        else:
            cleanup['settings_restored']=True
    report=summarize(rows,mode,status,abort_reason,collision_events)
    report['diagnostic_interpretation']=('Manual first gear steering trial; wheel angles are not frame locked and wheelbase is not calibrated' if mode=='manual_steering' else 'Control-only manual first gear comparison; automatic mode remains unvalidated' if mode=='manual_first' else None)
    report['failure_state_before_cleanup']=failure_state_at_cleanup
    cleanup_ok=bool(cleanup['sensor_destroy_returned_success'] and cleanup['vehicle_destroy_returned_success'])
    if rows:
        # Summarize samples that identify whether CARLA actually honored requested controls.
        pulse_rows = [r for r in rows if r.get('phase') in ('throttle','throttle_straight')]
        report['applied_control_diagnostics'] = {
            'last_warmup_control': warmup_control_readback,
            'first_throttle_readback': {k:v for k,v in pulse_rows[0].items() if k.startswith('applied_')} if pulse_rows else None,
            'last_throttle_readback': {k:v for k,v in pulse_rows[-1].items() if k.startswith('applied_')} if pulse_rows else None,
            'min_applied_throttle_during_throttle': min((r['applied_throttle'] for r in pulse_rows),default=None),
            'max_applied_brake_during_throttle': max((r['applied_brake'] for r in pulse_rows),default=None),
            'observed_gears_during_throttle': sorted({r['applied_gear'] for r in pulse_rows}),
            'note':'Readback is post-tick Vehicle.get_control(); not a frame-stamped physics snapshot.'}
        if mode == 'manual_steering':
            turn = [r for r in rows if r.get('phase')=='steer_pulse']
            report['steering_diagnostics'] = {
                'requested_steer_sign': args.steer_sign,
                'commanded_steer_normalized': .08*args.steer_sign,
                'mean_front_wheel_deg_range': [min(.5*(r['wheel_front_left_deg']+r['wheel_front_right_deg']) for r in turn),
                                                max(.5*(r['wheel_front_left_deg']+r['wheel_front_right_deg']) for r in turn)] if turn else None,
                'peak_abs_cross_track_m': max(abs(r['cross_track_m']) for r in rows),
                'final_cross_track_m':rows[-1]['cross_track_m'],
                'final_heading_change_deg':rows[-1]['heading_from_start_deg'],
                'applied_steer_range': [min(r['applied_steer'] for r in turn),max(r['applied_steer'] for r in turn)] if turn else None,
                'note':'Signs expressed in CommonRoad XY; CARLA normalized steering convention is not yet calibrated'}
            geom = config['estimated_wheel_geometry']
            if geom.get('available') and 2.0 < geom['wheelbase_from_xy_m'] < 4.5:
                report['kinematic_comparison_measured_geometry'] = kinematic_model_comparison(rows,geom['wheelbase_from_xy_m'])
    report.update({'warmup_control_commands_issued':warmup_control_count,
                   'drive_control_commands_issued':drive_control_count,
                   'brake_cleanup_attempted':brake_cleanup_attempted,
                   'carla_control_applied':bool(warmup_control_count or drive_control_count or brake_cleanup_attempted),
                   'cleanup':cleanup,'cleanup_ok':cleanup_ok,
                   'settings_restored':cleanup['settings_restored'],
                   'observation_sha256':obs['sha256'], 'source_observation_frame':obs['source_frame'],
                   'nominal_start_cr_xy':obs['body_center_cr_xy_m'],
                   'preflight':pre,'configuration':config,
                   'note':'CARLA motion calibration only: no obstacle, no Rulebook, no Scenic, no live replanning.'})
    if not cleanup_ok or not cleanup['settings_restored']:
        report['status']='cleanup_or_restore_failed'
        report['abort_reason']=report['abort_reason'] or 'Actor cleanup or world restore unconfirmed'
    save_results(args.out,rows,report)
    print('MILESTONE 6B-3:',report['status'],'reason:',report['abort_reason'])
    print('Collision count:',report['collision_count'],'cleanup:',cleanup)
    print('Output:',args.out/'summary.json')
    return 0 if report['status']=='completed' else 2


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observation',type=Path,default=DEFAULT_OBSERVATION)
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--inspect-server',action='store_true',help='Read-only Town01 and corridor checks')
    modes.add_argument('--execute-straight',action='store_true',help='CONTROL+TICKS: 4.2-s low-speed straight calibration')
    modes.add_argument('--execute-manual-first',action='store_true',help='CONTROL+TICKS: 4.2-s comparison using explicit first gear; no physics changes')
    modes.add_argument('--execute-steering',action='store_true',help='LEGACY automatic-gear steering; requires an automatic straight baseline')
    modes.add_argument('--execute-manual-steering',action='store_true',help='CONTROL+TICKS: manual-first low-speed steering pulse after manual-first baseline')
    p.add_argument('--confirm-drive',action='store_true',help='Explicit acknowledgement required for either driving mode')
    p.add_argument('--baseline-report',type=Path,help='Completed matching straight or manual-first summary.json required for steering')
    p.add_argument('--repeat-baseline-report',type=Path,help='Optional second matching manual-first baseline for reproducibility check')
    p.add_argument('--steer-sign',type=int,choices=(-1,1),default=1)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=2000)
    p.add_argument('--timeout',type=float,default=10.)
    args=p.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout<=0:
        p.error('Timeout must be positive finite')
    driving=args.execute_straight or args.execute_steering or args.execute_manual_first or args.execute_manual_steering
    if driving != args.confirm_drive:
        p.error('Driving requires both an execute flag and --confirm-drive; confirmation is not valid otherwise')
    if (args.execute_steering or args.execute_manual_steering) and args.baseline_report is None:
        p.error('Steering requires --baseline-report from previously completed straight drive')
    if not (args.execute_steering or args.execute_manual_steering) and args.baseline_report is not None:
        p.error('--baseline-report is only for steering modes')
    if args.repeat_baseline_report is not None and not args.execute_manual_steering:
        p.error('--repeat-baseline-report is only for --execute-manual-steering')
    try:
        obs=read_stable_observation(args.observation)
        print('MILESTONE 6B-3: stable reference frame:',obs['source_frame'])
        print('Recorded nominal start:',obs['body_center_cr_xy_m'],'yaw:',obs['yaw_cr_rad'])
        if not driving and not args.inspect_server:
            print('OFFLINE PREVIEW: no CARLA connected, no actors, no controls or ticks')
            return 0
        if args.execute_steering or args.execute_manual_steering:
            expected = 'manual_first' if args.execute_manual_steering else 'straight'
            baseline = validate_baseline(args.baseline_report,obs['sha256'],expected_mode=expected)
            if args.repeat_baseline_report is not None:
                repeat = validate_baseline(args.repeat_baseline_report,obs['sha256'],expected_mode='manual_first')
                a,b=baseline,repeat
                dv=abs(a['measured_speed_response']['peak_forward_speed_mps']-b['measured_speed_response']['peak_forward_speed_mps'])
                dx=abs(a['max_displacement_m']-b['max_displacement_m'])
                if dv>0.30 or dx>0.50:
                    raise ValueError('Two manual-first baselines differ too much: peak speed delta %.3fm/s, displacement delta %.3fm' %(dv,dx))
                print('REPEATABILITY: manual-first peak speed delta %.4fm/s; displacement delta %.4fm' %(dv,dx))
        if driving:
            if (args.out/'summary.json').exists() or (args.out/'calibration_samples.csv').exists():
                raise FileExistsError('Refusing to overwrite a previous calibration result')
            return drive(args,obs,'straight' if args.execute_straight else 'manual_first' if args.execute_manual_first else 'manual_steering' if args.execute_manual_steering else 'steering')
        carla,client,world,bp,wp_t,info=preflight(args,obs,drive=False)
        print('READ-ONLY TOWN01 PREFLIGHT:',json.dumps(info,indent=2))
        print('No actors, controls, ticks or world-setting changes issued')
        return 0
    except (OSError,ValueError,RuntimeError,KeyError,TypeError) as exc:
        print('MILESTONE 6B-3 FAILED SAFE:',type(exc).__name__+': '+str(exc))
        return 2


if __name__=='__main__':
    raise SystemExit(main())
