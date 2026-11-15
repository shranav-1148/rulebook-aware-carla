"""Milestone 5C: longitudinal position feedback for CARLA 0.9.16 tracking.

Consumes Milestone 4E-B *_bicycle_samples.csv (CommonRoad frame) and drives
one spawned CARLA vehicle with VehicleControl. Default is safe *offline preview*;
--inspect-server is read-only; --execute is required to spawn/tick the world.

This tests TRACKING a fixed offline path, not Rulebook decision-making in CARLA.
This is a standalone derivative of the AI-authored Milestone 5A tracker;
the 5A script remains unchanged for historical baseline comparisons.
The offline 'obstacle' is NOT spawned. CARLA and CommonRoad conventions differ
by y reflection and yaw sign. There are no per-tick teleports/velocity overrides.
"""
from __future__ import annotations

import argparse
import bisect
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
DEFAULT_TRAJ = HERE / 'output/town01_bicycle_graph_beam800_min40/corridor_first_bicycle_samples.csv'


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


@dataclass(frozen=True)
class Reference:
    t: float
    x: float            # CommonRoad BODY center
    y: float
    yaw: float          # CommonRoad radians
    v: float
    steering_deg: float
    rear_x: float       # CommonRoad REAR axle, for pure-pursuit target
    rear_y: float
    station: float
    offset: float


def load_references(path: Path) -> List[Reference]:
    required = ('t_s', 'x_m', 'y_m', 'yaw_rad', 'speed_mps',
                'steering_deg', 'rear_axle_x_m', 'rear_axle_y_m',
                'station_m', 'offset_m')
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or any(c not in reader.fieldnames for c in required):
            raise ValueError('Not a Milestone 4E-B dense bicycle CSV; required: '+', '.join(required))
        result = []
        for n, row in enumerate(reader, 2):
            try:
                nums = [float(row[k]) for k in required]
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Bad numeric value at CSV line {n}') from exc
            if not all(math.isfinite(v) for v in nums):
                raise ValueError(f'Nonfinite value at CSV line {n}')
            if nums[4] < 0:
                raise ValueError(f'Negative speed at CSV line {n}')
            result.append(Reference(*nums))
    if len(result) < 2 or abs(result[0].t) > 1e-8:
        raise ValueError('Expected >=2 points and an initial t=0 sample')
    if any(b.t-a.t <= 1e-7 for a,b in zip(result,result[1:])):
        raise ValueError('Time column must increase strictly')
    if result[-1].t > 300:
        raise ValueError('Refuse over-300-second tracking for this prototype')
    if max(math.hypot(r.rear_x-r.x,r.rear_y-r.y) for r in result) > 5.0:
        raise ValueError('Unexpected rear-axle/body-center reference offset')
    return result


def interpolate(a: Reference, b: Reference, t: float) -> Reference:
    if b.t <= a.t:
        raise ValueError('Non-increasing sample times')
    f = clamp((t-a.t)/(b.t-a.t),0.,1.)
    mix = lambda x,y: x + f*(y-x)
    return Reference(t, mix(a.x,b.x),mix(a.y,b.y), wrap(a.yaw+f*wrap(b.yaw-a.yaw)),
                     mix(a.v,b.v),mix(a.steering_deg,b.steering_deg),
                     mix(a.rear_x,b.rear_x),mix(a.rear_y,b.rear_y),
                     mix(a.station,b.station),mix(a.offset,b.offset))


def at_time(refs: List[Reference], t: float) -> Reference:
    if t <= refs[0].t:
        return refs[0]
    if t >= refs[-1].t:
        return refs[-1]
    i = bisect.bisect_right([r.t for r in refs],t)
    return interpolate(refs[i-1],refs[i],t)


def to_carla_xy(x: float, y: float):
    return (x,-y)


def to_carla_yaw(yaw_cr_rad: float) -> float:
    return wrap(-yaw_cr_rad)


def to_commonroad_yaw(yaw_carla_rad: float) -> float:
    return wrap(-yaw_carla_rad)


def rotate_local(x: float, y: float, yaw: float):
    return (x*math.cos(yaw)-y*math.sin(yaw),
            x*math.sin(yaw)+y*math.cos(yaw))


def actor_origin_from_body_center(cx,cy,yaw,box_center_local_xy):
    dx,dy=rotate_local(box_center_local_xy[0],box_center_local_xy[1],yaw)
    return cx-dx,cy-dy


def body_center_from_actor_origin(ax,ay,yaw,box_center_local_xy):
    dx,dy=rotate_local(box_center_local_xy[0],box_center_local_xy[1],yaw)
    return ax+dx,ay+dy


def pure_pursuit_angle(rear_x, rear_y, yaw, target_x, target_y, wheelbase):
    dx,dy=target_x-rear_x,target_y-rear_y
    length=max(0.25,math.hypot(dx,dy))
    alpha=wrap(math.atan2(dy,dx)-yaw)
    return math.atan2(2*wheelbase*math.sin(alpha),length)


def steering_curve_fraction(curve, speed_kmh):
    """CARLA VehiclePhysicsControl.steering_curve: x=speed km/h, y=steer fraction."""
    pairs=sorted((float(p.x),float(p.y)) for p in curve)
    if not pairs:
        return 1.0
    if speed_kmh<=pairs[0][0]:
        return clamp(pairs[0][1],0.05,1.0)
    for a,b in zip(pairs,pairs[1:]):
        if speed_kmh<=b[0]:
            f=(speed_kmh-a[0])/max(b[0]-a[0],1e-9)
            return clamp(a[1]+f*(b[1]-a[1]),0.05,1.0)
    return clamp(pairs[-1][1],0.05,1.0)


def target_at_lookahead(refs: List[Reference], rear_x: float, rear_y: float, lookahead_m: float):
    """Find closest spatial sample, then a future reference sample by traveled arc length.
    Simple 5A controller; not a robust tracker for looped/self-intersecting routes.
    """
    nearest=min(range(len(refs)), key=lambda i:(refs[i].rear_x-rear_x)**2+(refs[i].rear_y-rear_y)**2)
    dist=0.0
    for i in range(nearest+1,len(refs)):
        a,b=refs[i-1],refs[i]
        dist+=math.hypot(b.rear_x-a.rear_x,b.rear_y-a.rear_y)
        if dist >= lookahead_m:
            return b
    return refs[-1]


def tracking_errors(ref: Reference, actual_center_cr_x: float, actual_center_cr_y: float, actual_yaw_cr: float):
    dx=actual_center_cr_x-ref.x
    dy=actual_center_cr_y-ref.y
    return {'position_error_m':math.hypot(dx,dy),
            'along_track_error_m':dx*math.cos(ref.yaw)+dy*math.sin(ref.yaw),
            'cross_track_error_m':-dx*math.sin(ref.yaw)+dy*math.cos(ref.yaw),
            'heading_error_deg':math.degrees(wrap(actual_yaw_cr-ref.yaw))}


def command_for_state(refs, t, rear_x, rear_y, yaw_carla, speed_mps, wheelbase_m,
                      front_wheel_max_angle_deg, steer_curve, previous_steer,
                      fixed_dt, lookahead_m, max_steer_normalized=0.8):
    """Normalized steering, throttle and brake; CARLA frame (positive right yaw)."""
    target=target_at_lookahead(refs,rear_x,rear_y,lookahead_m)
    target_xy=to_carla_xy(target.rear_x,target.rear_y)
    delta=pure_pursuit_angle(rear_x,rear_y,yaw_carla,*target_xy,wheelbase_m)
    wheel_limit=max(1.,front_wheel_max_angle_deg*steering_curve_fraction(steer_curve,speed_mps*3.6))
    desired=clamp(math.degrees(delta)/wheel_limit,-max_steer_normalized,max_steer_normalized)
    # Limit abrupt normalized steering updates; not a full actuator model.
    steer=clamp(desired,previous_steer-1.5*fixed_dt,previous_steer+1.5*fixed_dt)
    planned=at_time(refs,t)
    err=planned.v-speed_mps
    if err < -0.5:
        throttle,brake=0.,clamp(0.10+0.18*(-err),0.,0.6)
    else:
        throttle,brake=clamp(0.28+0.23*err,0.,0.8),0.
    return steer,throttle,brake,target



def longitudinal_feedback(ref: Reference, actual_x_cr: float, actual_y_cr: float,
                          speed_mps: float, *, position_kp: float = 0.60,
                          max_speed_correction: float = 2.0,
                          max_commanded_speed: float = 12.0):
    """Return feedback controls and diagnostics, in CommonRoad coordinates.

    error sign: along-track actual-minus-reference is positive when ahead;
    lag = -along-track error is positive when behind the *time-indexed* ref.
    The target-speed correction is saturated and can never request reverse.
    The original 5A speed-to-throttle/brake law is retained to isolate the
    contribution of the added position-feedback term in the first experiment.
    This is not a calibrated automotive controller or guaranteed improvement.
    """
    if not (0.0 <= position_kp <= 3.0 and 0.0 <= max_speed_correction <= 5.0
            and 0.0 < max_commanded_speed <= 30.0):
        raise ValueError('Invalid bounded longitudinal controller gains')
    vals=(ref.x, ref.y, ref.yaw, ref.v, actual_x_cr, actual_y_cr, speed_mps)
    if not all(math.isfinite(v) for v in vals):
        raise ValueError('Nonfinite controller input')
    if speed_mps < 0.0:
        raise ValueError('Speed must be nonnegative')
    along_error=((actual_x_cr-ref.x)*math.cos(ref.yaw)
                 +(actual_y_cr-ref.y)*math.sin(ref.yaw))
    lag=-along_error
    correction=clamp(position_kp*lag,-max_speed_correction,max_speed_correction)
    target_speed=clamp(ref.v+correction,0.0,max_commanded_speed)
    speed_error=target_speed-speed_mps
    if speed_error < -0.5:
        throttle=0.0
        brake=clamp(0.10+0.18*(-speed_error),0.0,0.6)
    else:
        throttle=clamp(0.28+0.23*speed_error,0.0,0.8)
        brake=0.0
    return {'throttle':throttle,'brake':brake,'along_lag_m':lag,
            'speed_correction_mps':correction,
            'commanded_speed_mps':target_speed,
            'speed_error_mps':speed_error}

def summarize_rows(rows,collisions,status,abort_reason=None):
    if not rows:
        return {'status':status,'abort_reason':abort_reason,'samples':0,'collision_events':collisions}
    along_rmse=(math.sqrt(sum(r['along_track_error_m']**2 for r in rows)/len(rows))
                if all('along_track_error_m' in r for r in rows) else None)
    speed_rmse=(math.sqrt(sum((r['ref_speed_mps']-r['actual_speed_mps'])**2 for r in rows)/len(rows))
                if all('ref_speed_mps' in r and 'actual_speed_mps' in r for r in rows) else None)
    return {'status':status,'abort_reason':abort_reason,'samples':len(rows),
            'duration_s':rows[-1]['t_s'],
            'position_rmse_m':math.sqrt(sum(r['position_error_m']**2 for r in rows)/len(rows)),
            'mean_position_error_m':sum(r['position_error_m'] for r in rows)/len(rows),
            'max_position_error_m':max(r['position_error_m'] for r in rows),
            'final_position_error_m':rows[-1]['position_error_m'],
            'max_abs_cross_track_error_m':max(abs(r['cross_track_error_m']) for r in rows),
            'max_abs_heading_error_deg':max(abs(r['heading_error_deg']) for r in rows),
            'final_speed_mps':rows[-1]['actual_speed_mps'],
            'along_track_rmse_m':along_rmse,
            'final_signed_along_track_error_m':rows[-1].get('along_track_error_m'),
            'speed_tracking_rmse_mps':speed_rmse,
            'collision_event_count':len(collisions), 'collision_events':collisions}


def save_rows(path: Path, rows):
    if not rows:
        return
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def make_plot(out: Path, rows):
    if not rows:
        return False
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot([r['ref_x_cr_m'] for r in rows],[r['ref_y_cr_m'] for r in rows],label='Offline planned (CommonRoad)')
    ax.plot([r['actual_x_cr_m'] for r in rows],[r['actual_y_cr_m'] for r in rows],label='CARLA executed (mirrored to CommonRoad)')
    ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('CommonRoad X (m)');ax.set_ylabel('CommonRoad Y (m)')
    ax.set_title('Milestone 5C: planned versus executed body-center path')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    fig.savefig(out/'tracking_overlay.png',dpi=150);plt.close(fig)
    return True



def make_longitudinal_plot(out: Path, rows):
    if not rows:
        return False
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    t=[r['t_s'] for r in rows]
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
    axes[0].plot(t,[r['along_track_error_m'] for r in rows],label='Actual - planned along-track (m)')
    axes[0].axhline(0,color='gray',linewidth=0.75)
    axes[0].set_ylabel('Signed along error (m)');axes[0].grid(alpha=.25);axes[0].legend()
    axes[1].plot(t,[r['ref_speed_mps'] for r in rows],label='Planned speed')
    axes[1].plot(t,[r['actual_speed_mps'] for r in rows],label='Actual speed')
    axes[1].plot(t,[r['commanded_speed_mps'] for r in rows],label='Position-corrected target',linestyle='--')
    axes[1].set(xlabel='Simulation time (s)',ylabel='Speed (m/s)')
    axes[1].grid(alpha=.25);axes[1].legend()
    fig.suptitle('Milestone 5C: longitudinal tracking diagnostics')
    fig.tight_layout();fig.savefig(out/'longitudinal_tracking.png',dpi=150);plt.close(fig)
    return True

def get_actual(vehicle,carla_box_offset_xy,carla):
    transform=vehicle.get_transform()
    yaw=math.radians(transform.rotation.yaw)
    cx,cy=body_center_from_actor_origin(transform.location.x,transform.location.y,yaw,carla_box_offset_xy)
    vel=vehicle.get_velocity()
    speed=math.sqrt(vel.x*vel.x+vel.y*vel.y+vel.z*vel.z)
    return {'x_carla':cx,'y_carla':cy,'yaw_carla':yaw,'speed_mps':speed,
            'x_cr':cx,'y_cr':-cy,'yaw_cr':to_commonroad_yaw(yaw)}


def make_log_row(t,ref,actual,steer,throttle,brake,world_frame,diagnostics=None):
    e=tracking_errors(ref,actual['x_cr'],actual['y_cr'],actual['yaw_cr'])
    if diagnostics is None:
        diagnostics={'along_lag_m':-e['along_track_error_m'],
                     'speed_correction_mps':0.0,
                     'commanded_speed_mps':ref.v,
                     'speed_error_mps':ref.v-actual['speed_mps']}
    # Controls were calculated before the tick; current position error is
    # logged after the tick. Keep their time origins unambiguous.
    control_input_lag=diagnostics['along_lag_m']
    diagnostics={k:diagnostics[k] for k in ('speed_correction_mps',
                                              'commanded_speed_mps','speed_error_mps')}
    diagnostics['control_input_lag_m']=control_input_lag
    diagnostics['along_lag_m']=-e['along_track_error_m']
    return {'t_s':round(t,7),'world_frame':world_frame,
            'ref_x_cr_m':ref.x,'ref_y_cr_m':ref.y,
            'ref_yaw_cr_deg':math.degrees(ref.yaw),'ref_speed_mps':ref.v,
            'ref_station_m':ref.station,'ref_offset_m':ref.offset,
            'actual_x_cr_m':actual['x_cr'],'actual_y_cr_m':actual['y_cr'],
            'actual_yaw_cr_deg':math.degrees(actual['yaw_cr']),
            'actual_speed_mps':actual['speed_mps'],
            'actual_x_carla_m':actual['x_carla'],'actual_y_carla_m':actual['y_carla'],
            'control_steer':steer,'control_throttle':throttle,'control_brake':brake,**e,**diagnostics}


def server_preflight(args,refs,spawn=False):
    try:
        import carla
    except ImportError as exc:
        raise RuntimeError('CARLA Python API is not installed in this interpreter. Use Windows .venv with CARLA 0.9.16.') from exc
    client=carla.Client(args.host,args.port)
    client.set_timeout(args.timeout)
    world=client.get_world()
    world_map=world.get_map()
    client_version=str(client.get_client_version())
    server_version=str(client.get_server_version())
    print('CARLA client/server:',client_version, '/',server_version)
    if '0.9.16' not in client_version or '0.9.16' not in server_version:
        raise RuntimeError('This prototype targets CARLA 0.9.16 on both client and server. Verify compatible versions before execution.')
    print('Loaded map:',world_map.name)
    if 'Town01' not in world_map.name.split('/')[-1]:
        raise RuntimeError('Refusing to run: expected CARLA Town01 loaded, not '+world_map.name)
    try:
        bp=world.get_blueprint_library().find(args.blueprint)
    except Exception as exc:
        raise RuntimeError(f'Vehicle blueprint not available: {args.blueprint}') from exc
    cx,cy=to_carla_xy(refs[0].x,refs[0].y)
    waypoint=world_map.get_waypoint(carla.Location(cx,cy,0.),project_to_road=True,lane_type=carla.LaneType.Driving)
    if waypoint is None:
        raise RuntimeError('Could not find a driving-lane waypoint at the reference start')
    wp_t=waypoint.transform
    lateral=math.hypot(wp_t.location.x-cx,wp_t.location.y-cy)
    yaw=to_carla_yaw(refs[0].yaw)
    yaw_diff=abs(math.degrees(wrap(yaw-math.radians(wp_t.rotation.yaw))))
    print('Reference start CARLA (body center): (%.3f, %.3f), yaw=%.2f deg, speed=%.2f m/s' %
          (cx,cy,math.degrees(yaw),refs[0].v))
    print('Nearest road waypoint: center distance=%.3fm, heading difference=%.2fdeg, Z=%.3fm' %
          (lateral,yaw_diff,wp_t.location.z))
    if lateral>args.max_spawn_road_distance or yaw_diff>args.max_spawn_heading_diff:
        raise RuntimeError('Starting point appears inconsistent with CARLA Town01 lane. Stop and verify coordinate registration.')
    current_vehicles=list(world.get_actors().filter('vehicle.*'))
    print('Existing vehicles:',len(current_vehicles))
    if spawn and current_vehicles and not args.permit_existing_vehicles:
        raise RuntimeError('World has existing vehicles; refusing to take over simulation ticks. Use an isolated Town01 session.')
    if spawn and world.get_settings().synchronous_mode and not args.permit_existing_sync:
        raise RuntimeError('World already synchronous. Another client may be tick owner; refusing by default.')
    return carla,world,wp_t,bp,client


def execute(args,refs):
    carla,world,wp_t,bp,client=server_preflight(args,refs,spawn=True)
    original_settings=world.get_settings()
    settings_changed=False
    ego=None
    collision_sensor=None
    collision_events=[]
    rows=[]
    abort_reason=None
    status='aborted'
    metadata={}
    try:
        cfg=world.get_settings()
        cfg.synchronous_mode=True
        cfg.fixed_delta_seconds=args.fixed_dt
        world.apply_settings(cfg)
        settings_changed=True
        yaw=to_carla_yaw(refs[0].yaw)
        cx,cy=to_carla_xy(refs[0].x,refs[0].y)
        # Initial provisional placement. Actor bounding-box center isn't known
        # until after spawn; one pre-run correction happens BEFORE any tracking ticks.
        transform=carla.Transform(carla.Location(cx,cy,wp_t.location.z+args.spawn_z_offset),
                                  carla.Rotation(yaw=math.degrees(yaw)))
        if bp.has_attribute('role_name'):
            bp.set_attribute('role_name','milestone5c_ego')
        ego=world.try_spawn_actor(bp,transform)
        if ego is None:
            raise RuntimeError('Vehicle spawn was blocked (collision or invalid placement). No tracking performed.')
        ego.set_autopilot(False)
        box_loc=ego.bounding_box.location
        bb_xy=(float(box_loc.x),float(box_loc.y))
        ox,oy=actor_origin_from_body_center(cx,cy,yaw,bb_xy)
        # Only ONE initial pose adjustment, never teleporting during tracking.
        ego.set_transform(carla.Transform(carla.Location(ox,oy,wp_t.location.z+args.spawn_z_offset),
                                          carla.Rotation(yaw=math.degrees(yaw))))
        ego.set_simulate_physics(True)
        sensor_bp=world.get_blueprint_library().find('sensor.other.collision')
        collision_sensor=world.spawn_actor(sensor_bp,carla.Transform(),attach_to=ego)
        def on_collision(ev):
            collision_events.append({'frame':int(ev.frame),
                                     'other_actor_id':int(ev.other_actor.id),
                                     'other_actor_type':str(ev.other_actor.type_id)})
        collision_sensor.listen(on_collision)
        # ONE-TIME initial velocity assignment, not a per-tick control shortcut.
        # This aims to match offline initial speed of 6m/s at t=0.
        if args.initialize_velocity:
            ego.set_target_velocity(carla.Vector3D(refs[0].v*math.cos(yaw),refs[0].v*math.sin(yaw),0.))
        world.tick()
        metadata={'carla_client_version':str(client.get_client_version()),
                  'carla_server_version':str(client.get_server_version()),
                  'vehicle_blueprint':args.blueprint,'actor_id':ego.id,
                  'bbox_center_local_x_m':bb_xy[0],'bbox_center_local_y_m':bb_xy[1],
                  'wheelbase_assumed_m':args.wheelbase,'initial_velocity_set_once':args.initialize_velocity,
                  'synthetic_obstacle_spawned':False}
        physics=ego.get_physics_control()
        max_front_wheel_angle=max(float(w.max_steer_angle) for w in physics.wheels[:2])
        steering_curve=physics.steering_curve
        if max_front_wheel_angle<=0:
            raise RuntimeError('Vehicle front wheels have no steering angle')
        actual=get_actual(ego,bb_xy,carla)
        e0=tracking_errors(refs[0],actual['x_cr'],actual['y_cr'],actual['yaw_cr'])
        if e0['position_error_m']>args.max_initial_error:
            raise RuntimeError('Actual initial position error %.3fm exceeds %.3fm' %
                               (e0['position_error_m'],args.max_initial_error))
        last_steer=0.
        rows.append(make_log_row(0.,refs[0],actual,0.,0.,0.,world.get_snapshot().frame))
        steps=int(math.ceil(refs[-1].t/args.fixed_dt))
        consecutive_failures=0
        for step in range(1,steps+1):
            tprev=(step-1)*args.fixed_dt
            # Rear axle in CARLA is inferred from body center and assumed wheelbase.
            yaw_now=actual['yaw_carla']
            rear_x=actual['x_carla']-args.wheelbase*0.5*math.cos(yaw_now)
            rear_y=actual['y_carla']-args.wheelbase*0.5*math.sin(yaw_now)
            steer,throttle,brake,_=command_for_state(
                refs,tprev,rear_x,rear_y,yaw_now,actual['speed_mps'],
                args.wheelbase,max_front_wheel_angle,steering_curve,last_steer,
                args.fixed_dt,args.lookahead)
            diagnostics=None
            if args.controller == 'position':
                planned_now=at_time(refs,tprev)
                diagnostics=longitudinal_feedback(
                    planned_now,actual['x_cr'],actual['y_cr'],actual['speed_mps'],
                    position_kp=args.position_kp,
                    max_speed_correction=args.max_speed_correction,
                    max_commanded_speed=min(args.max_commanded_speed,args.max_speed-0.5))
                throttle,brake=diagnostics['throttle'],diagnostics['brake']
            ego.apply_control(carla.VehicleControl(throttle=throttle,steer=steer,brake=brake))
            frame=world.tick()
            t=min(step*args.fixed_dt,refs[-1].t)
            actual=get_actual(ego,bb_xy,carla)
            ref=at_time(refs,t)
            row=make_log_row(t,ref,actual,steer,throttle,brake,frame,diagnostics)
            rows.append(row)
            last_steer=steer
            if collision_events:
                abort_reason='CARLA collision sensor event';break
            failed=(row['position_error_m']>args.max_tracking_error or
                    abs(row['heading_error_deg'])>args.max_tracking_heading_error or
                    actual['speed_mps']>args.max_speed)
            consecutive_failures=consecutive_failures+1 if failed else 0
            if consecutive_failures>=3:
                abort_reason='tracking error, heading, or speed safety cutoff';break
            if step % max(1,round(1.0/args.fixed_dt))==0:
                print('t=%4.1fs  error=%5.2fm  speed=%4.1fm/s  steer=%+.2f' %
                      (t,row['position_error_m'],actual['speed_mps'],steer))
        if abort_reason is None:
            status='completed'
    except KeyboardInterrupt:
        abort_reason='interrupted by user'
    except Exception as exc:
        abort_reason=type(exc).__name__+': '+str(exc)
        print('ERROR:',abort_reason)
    finally:
        if ego is not None:
            try:
                ego.apply_control(carla.VehicleControl(throttle=0.,steer=0.,brake=1.))
            except Exception:
                pass
        if collision_sensor is not None:
            try:
                collision_sensor.stop()
                collision_sensor.destroy()
            except Exception as exc:
                print('WARNING: sensor cleanup:',exc)
        if ego is not None:
            try:
                ego.destroy()
            except Exception as exc:
                print('WARNING: actor cleanup:',exc)
        if settings_changed:
            try:
                world.apply_settings(original_settings)
            except Exception as exc:
                print('WARNING: could not restore original world settings:',exc)
    args.out.mkdir(parents=True,exist_ok=True)
    save_rows(args.out/'tracking_samples.csv',rows)
    stats=summarize_rows(rows,collision_events,status,abort_reason)
    stats.update(metadata)
    stats.update({'source_csv':str(args.trajectory.resolve()),'source_sha256':hashlib.sha256(args.trajectory.read_bytes()).hexdigest(),
                  'fixed_dt_s':args.fixed_dt,'lookahead_m':args.lookahead,
                  'world_map':'Town01','controls_are_vehiclecontrol':True,
                  'controller_type':args.controller,
                  'position_kp_per_second':args.position_kp if args.controller=='position' else 0.0,
                  'max_speed_correction_mps':args.max_speed_correction if args.controller=='position' else 0.0,
                  'max_commanded_speed_mps':args.max_commanded_speed})
    stats['overlay_written']=make_plot(args.out,rows)
    stats['longitudinal_plot_written']=make_longitudinal_plot(args.out,rows)
    (args.out/'summary.json').write_text(json.dumps(stats,indent=2)+'\n',encoding='utf-8')
    print('\n===== MILESTONE 5C: CARLA TRACKING =====')
    print('status:',status,'reason:',abort_reason or 'none')
    print('logged samples:',len(rows),'tracking RMSE (m):',stats.get('position_rmse_m'))
    print('max position error (m):',stats.get('max_position_error_m'))
    print('along-track RMSE (m):',stats.get('along_track_rmse_m'))
    print('collision events:',len(collision_events))
    print('outputs:',args.out)
    if status!='completed':
        raise SystemExit(2)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trajectory',type=Path,default=DEFAULT_TRAJ)
    p.add_argument('--out',type=Path,default=HERE/'output/carla_tracking_5c/corridor_first')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--inspect-server',action='store_true',help='Read-only CARLA map/version/spawn preflight')
    mode.add_argument('--execute',action='store_true',help='Spawn vehicle and apply controls (changes world sync temporarily)')
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=2000)
    p.add_argument('--timeout',type=float,default=10.)
    p.add_argument('--blueprint',default='vehicle.tesla.model3')
    p.add_argument('--fixed-dt',type=float,default=.05)
    p.add_argument('--lookahead',type=float,default=4.)
    p.add_argument('--controller',choices=('position','legacy'),default='position',
                   help='Position adds time-indexed along-track feedback; legacy retains 5A longitudinal command')
    p.add_argument('--position-kp',type=float,default=0.60,
                   help='Convert behind-reference lag (m) into target-speed correction (m/s)')
    p.add_argument('--max-speed-correction',type=float,default=2.0)
    p.add_argument('--max-commanded-speed',type=float,default=12.0)
    p.add_argument('--overwrite-existing',action='store_true',
                   help='Explicitly allow overwriting an existing 5C run directory (not recommended)')
    p.add_argument('--wheelbase',type=float,default=2.7)
    p.add_argument('--spawn-z-offset',type=float,default=.5)
    p.add_argument('--max-spawn-road-distance',type=float,default=1.75)
    p.add_argument('--max-spawn-heading-diff',type=float,default=30.)
    p.add_argument('--max-initial-error',type=float,default=1.5)
    p.add_argument('--max-tracking-error',type=float,default=4.0)
    p.add_argument('--max-tracking-heading-error',type=float,default=60.)
    p.add_argument('--max-speed',type=float,default=15.)
    p.add_argument('--no-initialize-velocity',action='store_false',dest='initialize_velocity',
                   help='Start from rest instead of setting initial CARLA target velocity once')
    p.set_defaults(initialize_velocity=True)
    p.add_argument('--permit-existing-vehicles',action='store_true',
                   help='Override isolation check; not recommended for first run')
    p.add_argument('--permit-existing-sync',action='store_true',
                   help='Override existing sync-world check; ensure no other tick client')
    args=p.parse_args(argv)
    for name in ('fixed_dt','lookahead','wheelbase','timeout','max_spawn_road_distance','max_spawn_heading_diff',
                 'max_initial_error','max_tracking_error','max_tracking_heading_error','max_speed'):
        value=getattr(args,name)
        if not math.isfinite(value) or value<=0:
            p.error(name+' must be positive finite')
    if not (0 <= args.position_kp <= 3.0 and 0 <= args.max_speed_correction <= 5.0
            and 0 < args.max_commanded_speed <= args.max_speed-0.5):
        p.error('Invalid position controller limits; max-commanded-speed must be <= max-speed - 0.5')
    if args.fixed_dt>.1:
        p.error('fixed_dt must be <=0.1 for this prototype')
    if not args.trajectory.is_file():
        p.error('Trajectory CSV missing: '+str(args.trajectory))
    try:
        refs=load_references(args.trajectory)
    except ValueError as exc:
        p.error(str(exc))
    first,last=refs[0],refs[-1]
    print('===== MILESTONE 5C: TRAJECTORY PREFLIGHT =====')
    print('CSV:',args.trajectory)
    print('samples:',len(refs),'duration:',round(last.t,3),'s')
    print('CommonRoad body center start:',(round(first.x,3),round(first.y,3)))
    print('CARLA body center start:',tuple(round(v,3) for v in to_carla_xy(first.x,first.y)))
    print('CARLA start yaw (degrees):',round(math.degrees(to_carla_yaw(first.yaw)),3))
    print('Initial target speed:',round(first.v,3),'m/s')
    print('Longitudinal controller:',args.controller)
    print('Position feedback Kp:',args.position_kp,'/s; max speed correction:',args.max_speed_correction,'m/s')
    if args.execute and not args.overwrite_existing and any(
            (args.out/name).exists() for name in ('summary.json','tracking_samples.csv')):
        p.error('Refusing to overwrite an existing run. Select a new --out folder for each trial.')
    if args.inspect_server:
        server_preflight(args,refs,spawn=False)
        print('READ-ONLY INSPECTION COMPLETE: no vehicle spawned, no world ticks.')
    elif args.execute:
        execute(args,refs)
    else:
        print('DRY RUN ONLY: no CARLA connection, vehicle spawn, simulation tick, or control.')
        print('Next: --inspect-server (read-only); then --execute (drives in CARLA).')


if __name__=='__main__':
    main()
