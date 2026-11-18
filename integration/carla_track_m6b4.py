"""M6B-4: guarded low-speed EMPIRICAL path tracking, not Rulebook tracking.

Default: offline validation of prior 6B-3 +/- trials and fitted model.
--inspect-server: CARLA read-only.
--execute-track --confirm-drive: temporary Tesla in isolated Town01, manual first gear,
replays positive 6B-3 steering reference with small bounded feedback corrections.
No obstacle, Scenic, upstream planner, or autonomous replanning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

# Provides CARLA lifecycle, sampling, control, and persistence operations.
import carla_calibrate_m6b3 as b
# Model-specific validation remains available from the model module; common
# primitives come from the shared implementation.
from carla_model_m6b4 import read_trial, model_for_tracking
from carla_m6b_common import finite, DT, wrap, digest

HERE=Path(__file__).resolve().parent
DEFAULT_OBS=HERE/'output/carla_observation_6b2_stable/trial_003/live_observation.json'
DEFAULT_POS_SUMMARY=HERE/'output/carla_calibration_6b3/manual_steering_pos_001/summary.json'
DEFAULT_POS_SAMPLES=HERE/'output/carla_calibration_6b3/manual_steering_pos_001/calibration_samples.csv'
DEFAULT_MODEL=HERE/'output/carla_calibration_6b4/model_v1.json'
DEFAULT_OUT=HERE/'output/carla_calibration_6b4/tracking_positive_001'

# All tighter than original 6B-3 motion caps. Max steer never exceeds demonstrated 0.08.
MAX_STEER_NORM=.08
MAX_STEER_CORRECTION=.020
MAX_TRACK_ERROR_M=.20
MAX_TRACK_HEADING_ERROR_DEG=4.
MAX_SUCCESS_TRACK_ERROR_M=.12
MAX_SUCCESS_HEADING_ERROR_DEG=2.
MAX_SPEED_MPS=2.0
MAX_DISPLACEMENT_M=4.
MAX_CROSS_M=.25
MAX_HEADING_M=5.
MAX_PHYSICAL_WHEEL_DEG=10.
K_LATERAL_RAD_PER_M=.5
K_HEADING=0.6


def local_pose(row, start):
    """Position relative to start in start-yaw frame, in CommonRoad meters/radians."""
    yaw0=finite('start yaw',start['yaw_cr_rad'])
    dx=finite('x',row['x_cr_m'])-finite('x0',start['x_cr_m'])
    dy=finite('y',row['y_cr_m'])-finite('y0',start['y_cr_m'])
    return (dx*math.cos(yaw0)+dy*math.sin(yaw0),
            -dx*math.sin(yaw0)+dy*math.cos(yaw0),
            wrap(finite('yaw',row['yaw_cr_rad'])-yaw0))


def tracking_error(actual,actual_start,reference,reference_start):
    s,y,psi=local_pose(actual,actual_start)
    sr,yr,pr=local_pose(reference,reference_start)
    return {'station_error_m':s-sr,'lateral_error_m':y-yr,
            'heading_error_deg':math.degrees(wrap(psi-pr)),
            'position_error_m':math.hypot(s-sr,y-yr),
            'reference_station_m':sr,'reference_lateral_m':yr,
            'actual_station_m':s,'actual_lateral_m':y,
            'reference_heading_change_deg':math.degrees(pr),
            'actual_heading_change_deg':math.degrees(psi)}


def command_from_error(reference_row,err,model):
    """Small steering-only correction, derived from provisional measured steering gain.

    For positive CR lateral/yaw error, command positive CARLA steer to turn right.
    No changes to throttle/brake schedule, and no extrapolation above |0.08|.
    """
    gain=math.radians(finite('steering gain',model['front_wheel_signed_deg_per_normalized_steer']))
    sign=int(model['carla_command_to_commonroad_steering_sign'])
    if sign!=-1 or gain<=0:
        raise ValueError('Unexpected or unvalidated calibrated steering sign/gain')
    e_y=finite('lateral error',err['lateral_error_m'])
    e_psi=math.radians(finite('heading error',err['heading_error_deg']))
    target_cr_delta_correction=-K_LATERAL_RAD_PER_M*e_y-K_HEADING*e_psi
    correction=target_cr_delta_correction/(sign*gain)
    correction=max(-MAX_STEER_CORRECTION,min(MAX_STEER_CORRECTION,correction))
    th=finite('ref throttle',reference_row['command_throttle'])
    br=finite('ref brake',reference_row['command_brake'])
    ff=finite('ref steer',reference_row['command_steer_normalized'])
    steer=max(-MAX_STEER_NORM,min(MAX_STEER_NORM,ff+correction))
    if th<0 or th>.25+1e-9 or br<0 or br>.65+1e-9 or abs(ff)>.08+1e-9:
        raise ValueError('Reference commands exceed demonstrated limits')
    return (th,steer,br,ff,correction)


def tracking_abort_reasons(row):
    reasons=[]
    if row['position_error_m']>MAX_TRACK_ERROR_M:
        reasons.append('reference positional tracking-error bound')
    if abs(row['heading_error_deg'])>MAX_TRACK_HEADING_ERROR_DEG:
        reasons.append('reference heading tracking-error bound')
    if row['total_speed_mps']>MAX_SPEED_MPS or row['forward_speed_mps']<-.20:
        reasons.append('speed/reverse limit')
    if row['distance_from_start_m']>MAX_DISPLACEMENT_M:
        reasons.append('displacement limit')
    if abs(row['cross_track_m'])>MAX_CROSS_M:
        reasons.append('lateral motion limit')
    if abs(row['heading_from_start_deg'])>MAX_HEADING_M:
        reasons.append('heading motion limit')
    if abs(row['vertical_speed_mps'])>.5:
        reasons.append('unexpected vertical motion')
    if max(abs(row['wheel_front_left_deg']),abs(row['wheel_front_right_deg']))>MAX_PHYSICAL_WHEEL_DEG:
        reasons.append('unexpected front-wheel steering angle')
    return reasons


def ensure_server_isolation(world,owned_id):
    """Read-only check for other road users during run; called on every tenth tick."""
    actors=world.get_actors()
    vehicles=[a for a in actors.filter('vehicle.*') if int(a.id)!=int(owned_id)]
    walkers=list(actors.filter('walker.pedestrian.*'))
    if vehicles or walkers:
        raise RuntimeError('New road user detected during isolated tracking trial')


def check_expected_applied(row,th,steer,br):
    if (abs(row['applied_throttle']-th)>.02 or
        abs(row['applied_steer']-steer)>.02 or
        abs(row['applied_brake']-br)>.05 or
        row['applied_hand_brake'] or row['applied_reverse'] or
        row['applied_manual_gear_shift'] is not True or row['applied_gear']!=1):
        raise RuntimeError('Manual-first tracking applied-control mismatch')


def summary_for_rows(rows,status,reason,collisions,model_sha,reference,obs,pre,cleanup,extra):
    result={'milestone':'6B-4','mode':'empirical_steering_reference_tracking',
        'status':status,'abort_reason':reason,
        'source_observation_frame':obs['source_frame'],'observation_sha256':obs['sha256'],
        'model_sha256':model_sha,'positive_reference_summary_sha256':reference['summary_sha256'],
        'positive_reference_samples_sha256':reference['samples_sha256'],
        'carla_control_applied':extra.get('controls_issued',0)>0,
        'reference_source':'measured CARLA 6B-3 +0.08 manual-first steering run, NOT Rulebook',
        'rulebook_enabled':False,'scenic_enabled':False,'obstacle_spawned':False,
        'feedback_enabled':True,'manual_first_gear':True,
        'samples':len(rows),'collision_count':len(collisions),'collision_events':list(collisions),
        'limits':{'max_speed_mps':MAX_SPEED_MPS,'max_distance_m':MAX_DISPLACEMENT_M,
                  'max_lateral_m':MAX_CROSS_M,'max_heading_deg':MAX_HEADING_M,
                  'max_instantaneous_position_error_m':MAX_TRACK_ERROR_M,
                  'max_instantaneous_heading_error_deg':MAX_TRACK_HEADING_ERROR_DEG,
                  'max_abs_command_steer':MAX_STEER_NORM,
                  'max_abs_measured_wheel_angle_deg':MAX_PHYSICAL_WHEEL_DEG},
        'preflight':pre,'cleanup':cleanup,
        'cleanup_ok':cleanup.get('vehicle_destroy_returned_success',False) and cleanup.get('sensor_destroy_returned_success',False),
        'settings_restored':cleanup.get('settings_restored',False),
        'telemetry_limits':['get_control is last applied control, not independent transmission telemetry',
            'get_wheel_steer_angle is sampled outside WorldSnapshot and not frame-locked'],
        'additional_diagnostics':extra}
    if rows:
        result['max_time_aligned_position_error_m']=max(r['position_error_m'] for r in rows)
        result['max_abs_time_aligned_heading_error_deg']=max(abs(r['heading_error_deg']) for r in rows)
        result['final_position_error_m']=rows[-1]['position_error_m']
        result['final_heading_error_deg']=rows[-1]['heading_error_deg']
        result['max_measured_speed_mps']=max(r['total_speed_mps'] for r in rows)
        result['final_forward_speed_mps']=rows[-1]['forward_speed_mps']
        result['max_abs_cross_track_m']=max(abs(r['cross_track_m']) for r in rows)
        result['max_abs_steer_command']=max(abs(r['command_steer_normalized']) for r in rows)
    return result


def execute(args,obs,reference,model):
    carla,client,world,bp,wp_t,pre=b.preflight(args,obs,drive=True)
    original=world.get_settings()
    modified=False
    ego=sensor=None
    rows=[]; collisions=[]; controls=0; warmup=0; brake_attempted=False
    cleanup={'vehicle_destroy_returned_success':False,'sensor_destroy_returned_success':False,'settings_restored':False}
    status='aborted';reason=None
    refrows=reference['rows']
    try:
        settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=DT
        modified=True
        world.apply_settings(settings)
        x,y=obs['body_center_cr_xy_m']; yaw=-obs['yaw_cr_rad']
        if bp.has_attribute('role_name'):
            bp.set_attribute('role_name','milestone6b4_empirical_tracking')
        tentative=carla.Transform(carla.Location(x,-y,wp_t.location.z+.5),carla.Rotation(yaw=math.degrees(yaw)))
        ego=world.try_spawn_actor(bp,tentative)
        if ego is None:
            raise RuntimeError('Could not spawn owned Tesla Model 3')
        bb=ego.bounding_box.location
        ax,ay=b.actor_origin_for_center(x,-y,yaw,bb)
        ego.set_transform(carla.Transform(carla.Location(ax,ay,wp_t.location.z+.5),carla.Rotation(yaw=math.degrees(yaw))))
        ego.set_autopilot(False)
        ego.set_simulate_physics(True)
        sensor=world.spawn_actor(world.get_blueprint_library().find('sensor.other.collision'),carla.Transform(),attach_to=ego)
        if sensor is None:
            raise RuntimeError('Could not spawn collision sensor')
        sensor.listen(lambda e: collisions.append({'frame':int(e.frame),'other_actor_id':int(e.other_actor.id),
                                                     'other_actor_type':str(e.other_actor.type_id)}))
        settled=0;warm_start=None
        for i in range(80):
            ego.apply_control(carla.VehicleControl(throttle=0.,steer=0.,brake=.5))
            warmup+=1
            frame=b._get_tick(world,timeout_s=args.timeout)
            a=b.sample_actor(world,ego,carla,bb,expected_frame=frame)
            if warm_start is None: warm_start=a['t_sim_s']
            ok=abs(a['vertical_speed_mps'])<=.05 and math.hypot(a['forward_speed_mps'],a['lateral_speed_mps'])<=.05
            settled=settled+1 if ok else 0
            if collisions: raise RuntimeError('Collision during warmup')
            if a['t_sim_s']-warm_start>=.6 and settled>=10: break
        else:
            raise RuntimeError('Vehicle did not settle in four simulated seconds')
        start=b.sample_actor(world,ego,carla,bb)
        if (math.hypot(start['x_cr_m']-x,start['y_cr_m']-y)>.20 or
            abs(math.degrees(wrap(start['yaw_cr_rad']-obs['yaw_cr_rad'])))>2. or
            abs(start['forward_speed_mps'])>.05 or abs(start['vertical_speed_mps'])>.05):
            raise RuntimeError('Post-spawn pose/stability inconsistent with observation')
        pre['tracking_start_checked']=True
        reffirst=refrows[0]
        last_err=tracking_error(start,start,reffirst,reffirst)
        def record_sample(a,index,phase,th,steer,br,feedforward,correction):
            err=tracking_error(a,start,refrows[index],reffirst)
            record=b.enrich(a,start,(phase,th,steer,br),index*DT)
            record.update(b.applied_control_readback(ego))
            record.update(err)
            record['feedforward_steer_normalized']=feedforward
            record['feedback_steer_correction']=correction
            record['reference_t_s']=finite('ref t',refrows[index]['t_s'])
            return record
        rows.append(record_sample(start,0,'initial',0.,0.,0.,0.,0.))
        for j in range(1,85):
            ref=refrows[j]
            th,steer,br,ff,correction=command_from_error(ref,last_err,model)
            ego.apply_control(b.commanded_vehicle_control(carla,'manual_steering',th,steer,br))
            controls+=1
            frame=b._get_tick(world,timeout_s=args.timeout)
            a=b.sample_actor(world,ego,carla,bb,expected_frame=frame)
            row=record_sample(a,j,ref['phase'],th,steer,br,ff,correction)
            rows.append(row)
            if j>=2: check_expected_applied(row,th,steer,br)
            if collisions: raise RuntimeError('Collision sensor event')
            if j == 10 and max(r['forward_speed_mps'] for r in rows)<.05 and max(r['distance_from_start_m'] for r in rows)<.02:
                raise RuntimeError('No longitudinal motion during tracking throttle; aborting')
            bad=tracking_abort_reasons(row)
            if bad: raise RuntimeError('Tracking safety abort: '+', '.join(bad))
            if j==round(1.1/DT):
                if max(r['forward_speed_mps'] for r in rows)<.05 or max(r['distance_from_start_m'] for r in rows)<.02:
                    raise RuntimeError('No longitudinal motion before steering replay')
            if j % 10==0: ensure_server_isolation(world,ego.id)
            last_err=row
            if j%10==0:
                print('t=%.2fs speed=%.3fm/s ref_err=%.3fm yaw_err=%.3fdeg steer=%.3f phase=%s'%(
                    j*DT,row['forward_speed_mps'],row['position_error_m'],row['heading_error_deg'],steer,row['phase']))
        if max(abs(r['heading_from_start_deg']) for r in rows if r['phase']=='steer_pulse') < .1 or max(abs(r['wheel_front_left_deg']) for r in rows if r['phase']=='steer_pulse') < .25:
            raise RuntimeError('No measurable steering response during empirical reference tracking')
        if abs(rows[-1]['forward_speed_mps'])>.15:
            raise RuntimeError('Vehicle did not stop after brake phase')
        if max(r['position_error_m'] for r in rows)>MAX_SUCCESS_TRACK_ERROR_M:
            raise RuntimeError('Tracking completed but position error exceeds success threshold')
        if max(abs(r['heading_error_deg']) for r in rows)>MAX_SUCCESS_HEADING_ERROR_DEG:
            raise RuntimeError('Tracking completed but heading error exceeds success threshold')
        status='completed'
    except KeyboardInterrupt:
        reason='interrupted by user'
    except Exception as exc:
        reason=type(exc).__name__+': '+str(exc)
    # Cleanup path: always execute this block after the guarded operation to close resources or restore state.
    finally:
        if ego is not None:
            try:
                ego.apply_control(carla.VehicleControl(throttle=0.,steer=0.,brake=1.0))
                brake_attempted=True
                if modified: b._get_tick(world,timeout_s=args.timeout)
            except Exception as exc: cleanup['brake_tick_error']=str(exc)
        if sensor is not None:
            try:
                sensor.stop();cleanup['sensor_destroy_returned_success']=bool(sensor.destroy())
            except Exception as exc:cleanup['sensor_cleanup_error']=str(exc)
        else:cleanup['sensor_destroy_returned_success']=True
        if ego is not None:
            try:cleanup['vehicle_destroy_returned_success']=bool(ego.destroy())
            except Exception as exc:cleanup['vehicle_cleanup_error']=str(exc)
        else:cleanup['vehicle_destroy_returned_success']=True
        if modified:
            try:
                world.apply_settings(original)
                now=world.get_settings()
                cleanup['settings_restored']=(bool(now.synchronous_mode)==bool(original.synchronous_mode)
                                              and now.fixed_delta_seconds==original.fixed_delta_seconds)
            except Exception as exc:cleanup['settings_restore_error']=str(exc)
        else:cleanup['settings_restored']=True
    if collisions and status=='completed':
        status='aborted'
        reason='Late collision sensor event during final cleanup tick'
    if not cleanup['vehicle_destroy_returned_success'] or not cleanup['sensor_destroy_returned_success'] or not cleanup['settings_restored']:
        status='cleanup_or_restore_failed'
        reason=reason or 'CARLA owned actor cleanup or world restore not verified'
    result=summary_for_rows(rows,status,reason,collisions,digest(args.model),reference,obs,pre,cleanup,
                            {'controls_issued':controls,'warmup_control_commands_issued':warmup,
                             'emergency_brake_attempted':brake_attempted,
                             'gear':'manual_first','steering_gain_used':model['front_wheel_signed_deg_per_normalized_steer'],
                             'wheelbase_observed_m':model['geometric_wheelbase_m'],
                             'max_abs_normalized_steer_limit':MAX_STEER_NORM})
    output=b.save_results(args.out,rows,result)
    print('M6B-4 TRACK:',result['status'],'reason:',reason)
    print('Collision count:',len(collisions),'cleanup:',cleanup)
    print('Output:',output)
    return 0 if result['status']=='completed' else 2


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observation',type=Path,default=DEFAULT_OBS)
    p.add_argument('--positive-summary',type=Path,default=DEFAULT_POS_SUMMARY)
    p.add_argument('--positive-samples',type=Path,default=DEFAULT_POS_SAMPLES)
    p.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--inspect-server',action='store_true')
    modes.add_argument('--execute-track',action='store_true')
    p.add_argument('--confirm-drive',action='store_true')
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=2000)
    p.add_argument('--timeout',type=float,default=10.)
    args=p.parse_args(argv)
    if args.confirm_drive != args.execute_track:
        p.error('Only --execute-track --confirm-drive together permits simulated motion')
    if not math.isfinite(args.timeout) or args.timeout<=0:
        p.error('Timeout must be positive finite')
    try:
        obs=b.read_stable_observation(args.observation)
        reference=read_trial(args.positive_summary,args.positive_samples,1)
        if reference['summary']['observation_sha256']!=obs['sha256']:
            raise ValueError('Reference summary does not match stable observation bytes')
        model=model_for_tracking(args.model,obs['sha256'],reference['samples_sha256'],reference['summary_sha256'])
        print('M6B-4: validated stable observation, positive steering reference and provisional model')
        print('M6B-4: source = EMPIRICAL CARLA 6B-3 trace, NOT upstream Rulebook trajectory')
        if not args.inspect_server and not args.execute_track:
            print('OFFLINE ONLY: No CARLA connection, ticks, actors, or controls')
            return 0
        if args.execute_track and ((args.out/'summary.json').exists() or (args.out/'calibration_samples.csv').exists()):
            raise FileExistsError('Refusing to overwrite an existing tracking run')
        if args.inspect_server:
            *_,pre=b.preflight(args,obs,drive=False)
            print('READ ONLY PREFLIGHT:',json.dumps(pre,indent=2))
            return 0
        return execute(args,obs,reference,model)
    except (OSError,ValueError,RuntimeError,TypeError,KeyError,ImportError) as exc:
        print('M6B-4 FAILED SAFE:',type(exc).__name__+': '+str(exc))
        return 2

if __name__=='__main__':
    raise SystemExit(main())
