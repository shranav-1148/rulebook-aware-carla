"""Milestone 6B-3: inspect CARLA Model 3 physics without vehicle driving.

Explicitly spawn a temporary ego in an isolated asynchronous Town01 world; read
physics configuration; destroy only the actor created here. Never apply control,
change world settings, enable autopilot, or call world.tick()/wait_for_tick().
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from integration.carla_calibrate_m6b3 import preflight, read_stable_observation, actor_origin_for_center, DEFAULT_OBSERVATION

DEFAULT_OUT = Path(__file__).resolve().parent / 'output/carla_calibration_6b3/drivetrain_probe_001'


def number_or_none(value):
    try:
        v=float(value)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def vec2(v):
    return [number_or_none(v.x), number_or_none(v.y)]


def vec3(v):
    return [number_or_none(v.x), number_or_none(v.y), number_or_none(v.z)]


def physics_report(p):
    """Read-only serialization of vehicle physics. Missing optional fields are null."""
    get=lambda name: number_or_none(getattr(p,name,None))
    gears=[]
    for g in getattr(p,'forward_gears',[]):
        gears.append({'ratio':number_or_none(getattr(g,'ratio',None)),
                      'down_ratio':number_or_none(getattr(g,'down_ratio',None)),
                      'up_ratio':number_or_none(getattr(g,'up_ratio',None))})
    wheels=[]
    for w in getattr(p,'wheels',[]):
        pos=getattr(w,'position',None)
        wheels.append({'radius':number_or_none(getattr(w,'radius',None)),
                       'max_steer_angle_deg':number_or_none(getattr(w,'max_steer_angle',None)),
                       'tire_friction':number_or_none(getattr(w,'tire_friction',None)),
                       'position':vec3(pos) if pos is not None else None})
    return {'mass_kg':get('mass'),'max_rpm':get('max_rpm'),
            'use_gear_autobox':bool(p.use_gear_autobox) if hasattr(p,'use_gear_autobox') else None,
            'gear_switch_time_s':get('gear_switch_time'),
            'clutch_strength':get('clutch_strength'),
            'final_ratio':get('final_ratio'),
            'reverse_gear_ratio':get('reverse_gear_ratio'),
            'drag_coefficient':get('drag_coefficient'),
            'torque_curve_rpm_torque': [vec2(q) for q in getattr(p,'torque_curve',[])],
            'forward_gears':gears,'wheels':wheels}


def control_report(c):
    return {'throttle':number_or_none(c.throttle), 'brake':number_or_none(c.brake),
            'hand_brake':bool(c.hand_brake), 'reverse':bool(c.reverse),
            'manual_gear_shift':bool(c.manual_gear_shift), 'gear_field':int(c.gear)}


def write_report(out, report):
    path=Path(out)/'drivetrain_report.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return path


def run(args):
    if (Path(args.out)/'drivetrain_report.json').exists():
        raise FileExistsError('Refusing to overwrite prior probe output before spawning an actor')
    obs=read_stable_observation(args.observation)
    # Use the same stringent occupancy / map / waypoint conditions as the drive test.
    carla, client, world, bp, waypoint, preflight_report = preflight(args,obs,drive=True)
    if world.get_settings().synchronous_mode:
        raise RuntimeError('World became synchronous; no temporary actor created')
    report={'milestone':'6B-3','mode':'temporary_spawn_drivetrain_inspection',
            'status':'not_started','no_vehicle_control_applied':True,
            'no_client_ticks_issued':True,'no_world_settings_changed':True,
            'source_frame':obs['source_frame'], 'preflight':preflight_report,
            'cleanup':{'actor_destroy_returned_success':None},
            'note':'Physics config and last control fields are not measured drivetrain gear engagement.'}
    vehicle=None
    try:
        x,y=obs['body_center_cr_xy_m']; yaw=-obs['yaw_cr_rad']
        # Provisional Z avoids an intrusive ground-level spawn; no settling requested.
        initial=carla.Transform(carla.Location(x,-y,waypoint.location.z+0.5),
                                carla.Rotation(yaw=math.degrees(yaw)))
        vehicle=world.try_spawn_actor(bp,initial)
        if vehicle is None:
            raise RuntimeError('Cannot spawn temporary Tesla for physics inspection')
        report['actor_id']=int(vehicle.id)
        report['actor_type']=str(vehicle.type_id)
        report['physics']=physics_report(vehicle.get_physics_control())
        report['initial_last_control']=control_report(vehicle.get_control())
        try:
            report['failure_state']=str(vehicle.get_failure_state())
        except (AttributeError,RuntimeError):
            report['failure_state']='unavailable'
        report['status']='physics_read'
    except Exception as exc:
        report['status']='probe_failed'
        report['error']=type(exc).__name__+': '+str(exc)
    finally:
        if vehicle is not None:
            try:
                report['cleanup']['actor_destroy_returned_success']=bool(vehicle.destroy())
            except Exception as exc:
                report['cleanup']['actor_destroy_returned_success']=False
                report['cleanup']['error']=type(exc).__name__+': '+str(exc)
        else:
            report['cleanup']['actor_destroy_returned_success']=True
        if report['cleanup']['actor_destroy_returned_success'] is not True:
            report['status']='cleanup_unconfirmed'
        path=write_report(args.out,report)
        print('MILESTONE 6B-3 DRIVETRAIN PROBE:',report['status'])
        print('Autobox:',(report.get('physics') or {}).get('use_gear_autobox'))
        print('Gear switch time:',(report.get('physics') or {}).get('gear_switch_time_s'))
        print('Number of forward gears:',len((report.get('physics') or {}).get('forward_gears',[])))
        print('Torque curve points:',len((report.get('physics') or {}).get('torque_curve_rpm_torque',[])))
        print('Cleanup returned:',report['cleanup']['actor_destroy_returned_success'])
        print('Report:',path)
    return 0 if report['status']=='physics_read' else 2


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observation',type=Path,default=DEFAULT_OBSERVATION)
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--confirm-temporary-spawn',action='store_true',
                   help='Required because this inspection creates and destroys one actor')
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=2000)
    p.add_argument('--timeout',type=float,default=10.)
    args=p.parse_args(argv)
    if not args.confirm_temporary_spawn:
        p.error('Explicit --confirm-temporary-spawn required; this briefly creates an actor')
    if not math.isfinite(args.timeout) or args.timeout<=0:
        p.error('Invalid --timeout')
    try:
        return run(args)
    except Exception as exc:
        print('MILESTONE 6B-3 FAILED SAFE:',type(exc).__name__+': '+str(exc))
        return 2


if __name__=='__main__':
    raise SystemExit(main())
