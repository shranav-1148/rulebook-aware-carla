"""M6B-4: offline, provisional CARLA steering adapter from paired 6B-3 observations.

No CARLA import, no vehicle control. Does not modify upstream CommonRoad source.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

DT = 0.05
VERSION = 'm6b4-provisional-steering-v1'


def finite(name, v):
    try:
        n = float(v)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} is not numeric') from exc
    if not math.isfinite(n):
        raise ValueError(f'{name} is not finite')
    return n


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_trial(summary_path, samples_path, expected_sign):
    summary_path, samples_path = Path(summary_path), Path(samples_path)
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if (summary.get('status') != 'completed' or summary.get('mode') != 'manual_steering'
            or summary.get('collision_count') != 0 or summary.get('cleanup_ok') is not True
            or summary.get('settings_restored') is not True
            or summary.get('throttle_motion_gate',{}).get('passed') is not True
            or summary.get('steering_pulse_gate',{}).get('passed') is not True):
        raise ValueError('Reference must be completed safe manual_steering run')
    if int(summary.get('configuration',{}).get('steering_sign_command',0)) != expected_sign:
        raise ValueError('Steering command sign mismatch')
    if summary.get('configuration',{}).get('mode') != 'manual_steering':
        raise ValueError('Report configuration mode mismatch')
    if summary.get('preflight',{}).get('map') != 'Carla/Maps/Town01':
        raise ValueError('Expected Town01')
    if summary.get('preflight',{}).get('client_version') != '0.9.16' or summary['preflight'].get('server_version') != '0.9.16':
        raise ValueError('Expected CARLA 0.9.16')
    if not summary.get('observation_sha256'):
        raise ValueError('Missing source observation hash')
    if finite('max speed',summary['max_total_speed_mps']) > 3.5 or finite('final speed',summary['final_forward_speed_mps']) > .15:
        raise ValueError('Unsafe reference motion')
    with samples_path.open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=85 or summary['samples']!=len(rows):
        raise ValueError('Reference requires all 85 samples')
    required = ('frame','t_s','t_sim_s','x_cr_m','y_cr_m','yaw_cr_rad','forward_speed_mps',
                'wheel_front_left_deg','wheel_front_right_deg','applied_steer','applied_throttle',
                'applied_brake','command_steer_normalized','command_throttle','command_brake',
                'phase','applied_gear','applied_manual_gear_shift','applied_hand_brake','applied_reverse')
    if not all(k in rows[0] for k in required):
        raise ValueError('Incomplete reference CSV schema')
    for i,r in enumerate(rows):
        t=finite('time',r['t_s'])
        if abs(t-i*DT)>1.e-4:
            raise ValueError('Reference timestamps are not 0.05-second aligned')
        for k in ('t_sim_s','x_cr_m','y_cr_m','yaw_cr_rad','forward_speed_mps',
                  'wheel_front_left_deg','wheel_front_right_deg','applied_steer',
                  'applied_throttle','applied_brake','command_steer_normalized',
                  'command_throttle','command_brake'):
            finite(k,r[k])
        if i:
            if int(r['frame'])!=int(rows[i-1]['frame'])+1:
                raise ValueError('Reference frames are not consecutive')
            if abs((finite('sim time',r['t_sim_s'])-finite('previous sim time',rows[i-1]['t_sim_s']))-DT)>0.002:
                raise ValueError('Simulation timestamp gap or drift')
            phase,throttle,steer,brake = reference_command(i)
            if r['phase']!=phase or any(abs(finite(k,r[k])-v)>0.001 for k,v in (
                ('command_throttle',throttle),('command_steer_normalized',steer*expected_sign),('command_brake',brake))):
                raise ValueError('Unexpected reference control schedule')
            if (abs(finite('readback throttle',r['applied_throttle'])-throttle)>.02 or
                abs(finite('readback steering',r['applied_steer'])-steer*expected_sign)>.02 or
                abs(finite('readback brake',r['applied_brake'])-brake)>.05 or
                r['applied_manual_gear_shift']!='True' or int(r['applied_gear'])!=1 or
                r['applied_hand_brake']!='False' or r['applied_reverse']!='False'):
                raise ValueError('Reference control readback is not manual first gear')
    turn=[r for r in rows if r['phase']=='steer_pulse']
    if len(turn)!=14:
        raise ValueError('Reference must contain exactly 14 steering-pulse samples')
    signed_heading=finite('final heading change',summary['steering_diagnostics']['final_heading_change_deg'])
    if signed_heading*expected_sign>=-.1:
        raise ValueError('Heading sign not opposite CARLA steering sign')
    if abs(finite('final cross',summary['steering_diagnostics']['final_cross_track_m']))>.65:
        raise ValueError('Reference lateral limit exceeded')
    return {'summary':summary,'rows':rows,
            'summary_sha256':digest(summary_path),'samples_sha256':digest(samples_path)}


def reference_command(i):
    """Validated 6B-3 schedule; steering is unsigned before sign is applied."""
    if not 1 <= i <= 84:
        raise ValueError('Reference index outside 1..84')
    t=(i-1)*DT
    if t < 1.1-1e-9: return ('throttle_straight',.25,0.,0.)
    if t < 1.8-1e-9: return ('steer_pulse',.20,.08,0.)
    if t < 2.2-1e-9: return ('neutralize',0.,0.,0.)
    return ('brake',0.,0.,.65)


def command_model_residuals(rows, gain_deg_per_norm, wheelbase_m, steering_sign=-1, command_shift=0):
    """Predict kinematic response to commanded wheel angle, with MEASURED speed.

    command_shift +/-1 is a sensitivity comparison, not measured system latency.
    """
    if len(rows) < 2: raise ValueError('Insufficient samples')
    x=finite('x0',rows[0]['x_cr_m']); y=finite('y0',rows[0]['y_cr_m'])
    yaw=finite('yaw0',rows[0]['yaw_cr_rad'])
    position_err=[];heading_err=[]
    for i in range(1,len(rows)):
        previous,current=rows[i-1],rows[i]
        dt=finite('time',current['t_s'])-finite('prev time',previous['t_s'])
        if not 0.049 < dt < 0.051: raise ValueError('Invalid model integration interval')
        speed=max(0.,.5*(finite('v',previous['forward_speed_mps'])+finite('v',current['forward_speed_mps'])))
        src=rows[min(max(1,i+command_shift),len(rows)-1)]
        steer=finite('command steer',src['command_steer_normalized'])
        delta=math.radians(steering_sign*gain_deg_per_norm*steer)
        next_yaw=wrap(yaw+dt*speed*math.tan(delta)/wheelbase_m)
        mid_yaw=wrap(yaw+.5*wrap(next_yaw-yaw))
        x+=dt*speed*math.cos(mid_yaw)
        y+=dt*speed*math.sin(mid_yaw)
        yaw=next_yaw
        position_err.append(math.hypot(x-finite('rx',current['x_cr_m']),y-finite('ry',current['y_cr_m'])))
        heading_err.append(abs(math.degrees(wrap(yaw-finite('ryaw',current['yaw_cr_rad'])))))
    return {'position_rmse_m':math.sqrt(mean(z*z for z in position_err)),
            'heading_rmse_deg':math.sqrt(mean(z*z for z in heading_err))}


def estimate_model(positive, negative):
    ps,ns=positive['summary'],negative['summary']
    if ps['observation_sha256']!=ns['observation_sha256']:
        raise ValueError('Steering trials do not share the same stable observation')
    if positive['samples_sha256']==negative['samples_sha256']:
        raise ValueError('Identical sample file supplied twice')
    lp=ps['configuration']['estimated_wheel_geometry']
    ln=ns['configuration']['estimated_wheel_geometry']
    if not lp.get('available') or not ln.get('available'):
        raise ValueError('Both steering trials require measured axle geometry')
    wheelbases=[finite('wheelbase',v['wheelbase_from_xy_m']) for v in (lp,ln)]
    if max(wheelbases)-min(wheelbases)>.01 or not all(2.5<w<3.5 for w in wheelbases):
        raise ValueError('Inconsistent or implausible axle geometry')
    pulse = [(r,sign) for trial,sign in ((positive,1),(negative,-1))
             for r in trial['rows'] if r['phase']=='steer_pulse']
    angles=[.5*(finite('FL',r['wheel_front_left_deg'])+finite('FR',r['wheel_front_right_deg'])) for r,_ in pulse]
    if not all(a*sign>1. for a,(_,sign) in zip(angles,pulse)):
        raise ValueError('Physical wheel-steer sign does not match command sign')
    gains=[abs(a)/abs(finite('steer command',r['command_steer_normalized'])) for a,(r,_) in zip(angles,pulse)]
    gain=mean(gains)
    if not 40<gain<90:
        raise ValueError('Wheel steering response outside conservative observed range')
    timing={}
    for trial,label in ((positive,'positive'),(negative,'negative')):
        rows=trial['rows']; elapsed=[finite('sim time',r['t_sim_s']) for r in rows]
        gaps=[elapsed[i]-elapsed[i-1] for i in range(1,len(elapsed))]
        timing[label]={'frames_contiguous':True,'sample_count':len(rows),
                       'mean_sim_delta_s':mean(gaps),'max_abs_sim_delta_error_s':max(abs(d-DT) for d in gaps),
                       'physics_wheel_angles_frame_locked':False,
                       'control_readback_frame_locked_with_pose':False}
    response_audit={}
    for trial,label in ((positive,'positive'),(negative,'negative')):
        response_audit[label]={str(shift):command_model_residuals(trial['rows'],gain,mean(wheelbases),-1,shift) for shift in (-1,0,1)}
    return {
        'schema_version':VERSION,'status':'provisional_offline_fit',
        'carla_version':'0.9.16','carla_map':'Carla/Maps/Town01',
        'blueprint':'vehicle.tesla.model3',
        'observation_sha256':ps['observation_sha256'],
        'manual_gear':1,'automatic_mode_validated':False,
        'original_planner_wheelbase_m_unchanged':2.7,
        'geometric_wheelbase_m':mean(wheelbases),
        'carla_command_to_commonroad_steering_sign':-1,
        'front_wheel_signed_deg_per_normalized_steer':gain,
        'calibrated_steer_command_abs_max':.08,
        'calibration_speed_range_mps':[0.,max(finite('speed',r['forward_speed_mps']) for trial in (positive,negative) for r in trial['rows'])],
        'source_files':{
           'positive_summary_sha256':positive['summary_sha256'],'positive_samples_sha256':positive['samples_sha256'],
           'negative_summary_sha256':negative['summary_sha256'],'negative_samples_sha256':negative['samples_sha256']},
        'timing_audit':timing,
        'command_gain_kinematic_diagnostics':response_audit,
        'limitations':[
          'Measured front-wheel steering angles are queried separately after each snapshot; not frame-locked telemetry',
          'Applied-control readback is reported for the last tick, not independent drivetrain telemetry',
          'Geometric axle spacing is not a calibrated effective kinematic wheelbase',
          'Gain is local to normalized steering magnitude 0.08 and low vehicle speed',
          'Command-gain kinematic comparison uses measured speed; +/- one-sample shifts are sensitivity probes, not latency estimates',
          'No predictive throttle-to-speed or brake-to-speed dynamics fit',
          'Automatic gearbox remained immobile in a separate earlier run',
          'This adapter is not the original Rulebook planner and does not alter its code']}


def model_for_tracking(path, observation_hash, reference_sha, reference_summary_sha):
    model=json.loads(Path(path).read_text(encoding='utf-8'))
    if (model.get('schema_version')!=VERSION or model.get('status')!='provisional_offline_fit' or
        model.get('observation_sha256')!=observation_hash or model.get('manual_gear')!=1 or
        model.get('carla_command_to_commonroad_steering_sign')!=-1 or
        model.get('source_files',{}).get('positive_samples_sha256')!=reference_sha or
        model['source_files'].get('positive_summary_sha256')!=reference_summary_sha):
        raise ValueError('Calibration model does not match positive tracking reference and observation')
    if not 2.5<finite('wheelbase',model['geometric_wheelbase_m'])<3.5:
        raise ValueError('Invalid model wheelbase')
    if not 40<finite('steering gain',model['front_wheel_signed_deg_per_normalized_steer'])<90:
        raise ValueError('Invalid model steering gain')
    return model


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--positive-summary',type=Path,required=True)
    p.add_argument('--positive-samples',type=Path,required=True)
    p.add_argument('--negative-summary',type=Path,required=True)
    p.add_argument('--negative-samples',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    try:
        if args.out.exists():
            raise FileExistsError(f'Refusing to overwrite existing model {args.out}')
        pos=read_trial(args.positive_summary,args.positive_samples,1)
        neg=read_trial(args.negative_summary,args.negative_samples,-1)
        m=estimate_model(pos,neg)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(m,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print('M6B-4 OFFLINE MODEL:',m['status'])
        print('Wheelbase (geometric):',m['geometric_wheelbase_m'],'m')
        print('Steering gain:',m['front_wheel_signed_deg_per_normalized_steer'],'deg/normalized')
        print('Timing audit: wheel angles NOT snapshot-frame-locked')
        print('No CARLA connection or control. Output:',args.out)
        return 0
    except (ValueError,KeyError,TypeError,OSError) as exc:
        print('M6B-4 OFFLINE FIT FAILED SAFE:',type(exc).__name__+': '+str(exc))
        return 2

if __name__=='__main__':
    raise SystemExit(main())
