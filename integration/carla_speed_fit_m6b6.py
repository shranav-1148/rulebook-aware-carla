"""Milestone 6B-6 offline descriptive longitudinal response fit.

Uses TWO completed, distinct 6B-3 manual-first steering runs (positive/negative).
Outputs a PROVISIONAL empirical response envelope; not an engine/brake identification model.
Does not connect to CARLA or modify existing planner files.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median, mean
from carla_model_m6b4 import digest, finite, DT

HERE=Path(__file__).resolve().parent
POS=HERE/'output/carla_calibration_6b3/manual_steering_pos_001'
NEG=HERE/'output/carla_calibration_6b3/manual_steering_neg_001'
DEFAULT_OUT=HERE/'output/carla_calibration_6b6/longitudinal_fit_v1.json'
SCHEMA='m6b6-longitudinal-descriptive-v1'
PHASES=('throttle_straight','steer_pulse','neutralize','brake')

def read_trial(summary_path,samples_path,sign):
    summary_path=Path(summary_path);samples_path=Path(samples_path)
    report=json.loads(summary_path.read_text(encoding='utf-8'))
    if not (report.get('status')=='completed' and report.get('mode')=='manual_steering'
            and report.get('collision_count')==0 and report.get('cleanup_ok') is True
            and report.get('settings_restored') is True
            and report.get('steering_diagnostics',{}).get('requested_steer_sign')==sign
            and report.get('throttle_motion_gate',{}).get('passed') is True
            and report.get('steering_pulse_gate',{}).get('passed') is True):
        raise ValueError('Invalid completed manual-first calibration source')
    with samples_path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    if len(rows)!=85 or report.get('samples')!=85:
        raise ValueError('Expected 85 recorded samples in source')
    out=[]
    for i,row in enumerate(rows):
        if not all(k in row for k in ('frame','t_s','t_sim_s','forward_speed_mps','phase',
             'command_throttle','command_brake','applied_throttle','applied_brake',
             'applied_manual_gear_shift','applied_gear')):
            raise ValueError('Source CSV missing longitudinal control telemetry')
        t=finite('t_s',row['t_s']);sim=finite('t_sim_s',row['t_sim_s'])
        v=finite('speed',row['forward_speed_mps'])
        th=finite('throttle',row['command_throttle']);br=finite('brake',row['command_brake'])
        if (abs(t-i*DT)>1e-4 or not -.05<=v<=2.1 or not 0<=th<=.25+1e-6
                or not 0<=br<=.65+1e-6 or th*br>1e-6):
            raise ValueError('Invalid source timing, speed or controls')
        if i:
            prev=out[-1]
            if int(row['frame'])!=prev['frame']+1 or abs(sim-prev['sim']-DT)>.002:
                raise ValueError('Nonconsecutive frame/time in longitudinal trial')
            if (row['phase'] not in PHASES or row['applied_manual_gear_shift']!='True'
                    or int(row['applied_gear'])!=1
                    or abs(finite('applied throttle',row['applied_throttle'])-th)>.02
                    or abs(finite('applied brake',row['applied_brake'])-br)>.05):
                raise ValueError('Source control readback/gear mismatch')
        out.append({'frame':int(row['frame']),'sim':sim,'t':t,'v':v,
            'phase':row['phase'],'throttle':th,'brake':br})
    return {'rows':out,'report':report,'summary_sha256':digest(summary_path),
            'samples_sha256':digest(samples_path)}

def response(trials):
    if trials[0]['samples_sha256']==trials[1]['samples_sha256']:
        raise ValueError('Two DISTINCT steering experiment CSVs required')
    a,b=[t['report'] for t in trials]
    if (a.get('observation_sha256')!=b.get('observation_sha256') or
        a.get('configuration',{}).get('estimated_wheel_geometry',{}).get('wheelbase_from_xy_m')!=
        b.get('configuration',{}).get('estimated_wheel_geometry',{}).get('wheelbase_from_xy_m')):
        raise ValueError('Calibration trials have inconsistent source state/geometry')
    sets={p:[] for p in PHASES}
    for trial in trials:
        r=trial['rows']
        for prev,cur in zip(r,r[1:]):
            if cur['phase']!=prev['phase'] or cur['phase'] not in sets:
                continue  # exclude first tick after command transitions; lag still not identifiable
            accel=(cur['v']-prev['v'])/DT
            if cur['phase']=='brake' and not(cur['v']>.04 or prev['v']>.04):continue
            if cur['phase']=='neutralize' and prev['v']<.15:continue
            if cur['phase'] in ('throttle_straight','steer_pulse') and prev['v']<.03:continue
            sets[cur['phase']].append(accel)
    if min(len(sets[k]) for k in ('throttle_straight','steer_pulse','neutralize'))<4 or len(sets['brake'])<2:
        raise ValueError('Too few independent acceleration samples for descriptive model')
    stat={}
    for p,values in sets.items():
        stat[p]={'count':len(values),'mean_acceleration_mps2':mean(values),
                 'median_acceleration_mps2':median(values),
                 'min_acceleration_mps2':min(values),'max_acceleration_mps2':max(values)}
    # Strongly gated response sign; no claim linearity or fitted response dynamics.
    if stat['throttle_straight']['median_acceleration_mps2']<=.05 or stat['brake']['mean_acceleration_mps2']>=-.1:
        raise ValueError('Recorded throttle/brake responses do not support longitudinal feedback design')
    return {'schema_version':SCHEMA,'status':'descriptive_offline_fit','milestone':'6B-6',
        'observation_sha256':a['observation_sha256'],
        'positive_source_summary_sha256':trials[0]['summary_sha256'],
        'positive_source_samples_sha256':trials[0]['samples_sha256'],
        'negative_source_summary_sha256':trials[1]['summary_sha256'],
        'negative_source_samples_sha256':trials[1]['samples_sha256'],
        'data_source':'2 completed opposite-sign CARLA 6B-3 manual-first-gear steering trials',
        'phase_response_mps2':stat,
        'limits':{'speed_max_mps':2.0,'throttle_max':.25,'brake_max':.65,
                  'manual_first_gear_only':True},
        'limitations':['Descriptive finite-difference response, NOT a predictive drivetrain model',
          'Discrete 0.05s samples can contain gear, actuator, tire and braking transients',
          'First sample after phase transition excluded; only two useful interior brake differences across both runs',
          'Brake deceleration estimate is very sparse and may be highly nonlinear',
          'Throttle/brake causality, delays and speed dependence are not identified',
          'Calibration conducted only in low-speed manual first gear',
          'No original Rulebook planner or Scenic integration']}

def validate_fit(path,positive_summary=POS/'summary.json',positive_samples=POS/'calibration_samples.csv',
                 negative_summary=NEG/'summary.json',negative_samples=NEG/'calibration_samples.csv'):
    trials=[read_trial(positive_summary,positive_samples,1),
            read_trial(negative_summary,negative_samples,-1)]
    expected=response(trials)
    actual=json.loads(Path(path).read_text(encoding='utf-8'))
    if actual!=expected:raise ValueError('Longitudinal fit contents do not match source files')
    return actual

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--positive-summary',type=Path,default=POS/'summary.json')
    p.add_argument('--positive-samples',type=Path,default=POS/'calibration_samples.csv')
    p.add_argument('--negative-summary',type=Path,default=NEG/'summary.json')
    p.add_argument('--negative-samples',type=Path,default=NEG/'calibration_samples.csv')
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    a=p.parse_args(argv)
    try:
        trials=[read_trial(a.positive_summary,a.positive_samples,1),
                read_trial(a.negative_summary,a.negative_samples,-1)]
        doc=response(trials)
        if a.out.exists():raise FileExistsError('Refusing to overwrite an existing fit')
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print('M6B-6 OFFLINE longitudinal description:',json.dumps(doc['phase_response_mps2'],indent=2))
        print('NO CARLA connection, control or planner modification. Output:',a.out)
        return 0
    except (ValueError,TypeError,OSError,KeyError) as exc:
        print('M6B-6 OFFLINE FAILED SAFE:',type(exc).__name__+': '+str(exc));return 2

if __name__=='__main__':raise SystemExit(main())
