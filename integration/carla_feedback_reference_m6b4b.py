"""M6B-4B: OFFLINE, model-checked small perturbed empirical steering reference.

This is NOT a CommonRoad/Rulebook planned path. It applies the difference between
provisional-bicycle rollouts (same recorded speed, different steering) to the
measured positive-steering CARLA path. No CARLA API is imported or called.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

# Shared model validation and provenance helpers; model_for_tracking and
# read_trial remain exported by the model module for compatibility.
from carla_model_m6b4 import model_for_tracking, read_trial
from carla_m6b_common import DT, digest, finite, wrap
import carla_track_m6b4 as old

SCHEMA='m6b4b-perturbed-empirical-reference-v1'
PERTURB_AMPLITUDE=0.025  # normalized CARLA steering: reference construction only
PERTURB_RAMP_S=0.20
TARGET_LATERAL_MIN_M=0.020
TARGET_LATERAL_MAX_M=0.030


def integrate_commands(rows, commands, model):
    """Local bicycle rollout conditioned on *recorded* velocities, not a velocity prediction."""
    if len(rows)!=85 or len(commands)!=85:
        raise ValueError('Requires 85 reference samples/commands')
    wheelbase=finite('wheelbase',model['geometric_wheelbase_m'])
    gain=math.radians(finite('steering gain',model['front_wheel_signed_deg_per_normalized_steer']))
    sign=model['carla_command_to_commonroad_steering_sign']
    if sign!=-1 or not 2.5<wheelbase<3.5 or not 40<math.degrees(gain)<90:
        raise ValueError('Only validated provisional sign and wheelbase/gain ranges are supported')
    x=y=yaw=0.0
    states=[(x,y,yaw)]
    for j in range(1,len(rows)):
        vv=max(0.,.5*(finite('vprev',rows[j-1]['forward_speed_mps'])+finite('vcur',rows[j]['forward_speed_mps'])))
        if vv>2.0:
            raise ValueError('Reference speed exceeds tested tracking envelope')
        steer=finite('steer',commands[j])
        if abs(steer)>.08+1e-9:
            raise ValueError('Model steering demand outside demonstrated +/-0.08 range')
        delta=sign*gain*steer
        step=DT*vv
        dyaw=step*math.tan(delta)/wheelbase
        x+=step*math.cos(yaw+dyaw/2)
        y+=step*math.sin(yaw+dyaw/2)
        yaw=wrap(yaw+dyaw)
        states.append((x,y,yaw))
    return states


def generate_reference(trial,model,obs,model_hash):
    """Offset measured trace by *difference* of two model rollouts, not model absolute error."""
    rows=trial['rows']
    if (trial['summary']['observation_sha256']!=obs['sha256'] or
        model['observation_sha256']!=obs['sha256']):
        raise ValueError('Inconsistent stable observation')
    ff=[finite('empirical feedforward steer',r['command_steer_normalized']) for r in rows]
    altered=[ff[j]-PERTURB_AMPLITUDE*min(1.,j*DT/PERTURB_RAMP_S) for j in range(85)]
    base=integrate_commands(rows,ff,model)
    modified=integrate_commands(rows,altered,model)
    first=rows[0]
    targets=[]
    for j,r in enumerate(rows):
        station,lat,theta=old.local_pose(r,first)
        dm=modified[j]
        db=base[j]
        targets.append({
            'index':j,'t_s':round(j*DT,5),
            'reference_station_m':station+dm[0]-db[0],
            'reference_lateral_m':lat+dm[1]-db[1],
            'reference_heading_change_deg':math.degrees(wrap(theta+wrap(dm[2]-db[2]))),
            'reference_speed_input_mps':finite('speed',r['forward_speed_mps']),
            'baseline_feedforward_steer':ff[j],
            'feasible_model_steer':altered[j],
            'baseline_station_m':station,'baseline_lateral_m':lat,
            'baseline_heading_change_deg':math.degrees(theta),
            'perturbation_station_m':dm[0]-db[0],
            'perturbation_lateral_m':dm[1]-db[1],
            'perturbation_heading_deg':math.degrees(wrap(dm[2]-db[2])),
        })
    final=targets[-1]
    # Nontrivial model-checked offset; safety envelope is a *screen*, not proof of CARLA feasibility.
    if not TARGET_LATERAL_MIN_M<=abs(final['perturbation_lateral_m'])<=TARGET_LATERAL_MAX_M:
        raise ValueError('Perturbed reference did not reach requested 2-3 cm lateral difference')
    if (max(abs(r['perturbation_heading_deg']) for r in targets)>1.8 or
        max(abs(r['reference_lateral_m']) for r in targets)>.20 or
        max(abs(r['reference_heading_change_deg']) for r in targets)>4.0 or
        max(abs(r['feasible_model_steer']) for r in targets)>.08+1e-8):
        raise ValueError('Modified reference exceeds conservative motion screen')
    if not abs(final['reference_speed_input_mps'])<=.15:
        raise ValueError('Reference does not end at rest')
    return {
        'schema_version':SCHEMA, 'status':'offline_reference_generated',
        'reference_type':'empirical_CARLA_6B3_path_plus_provisional_bicycle_delta_NOT_Rulebook',
        'observation_sha256':obs['sha256'], 'observation_frame':obs['source_frame'],
        'model_sha256':model_hash,
        'positive_summary_sha256':trial['summary_sha256'],
        'positive_samples_sha256':trial['samples_sha256'],
        'reference_origin_cr':{
            'x_cr_m':finite('x0',first['x_cr_m']),
            'y_cr_m':finite('y0',first['y_cr_m']),
            'yaw_cr_rad':finite('yaw0',first['yaw_cr_rad'])},
        'dt_s':DT,'count':len(targets),
        'controller_feedforward_source':'unaltered_6B3_commands_in_both_comparison_arms',
        'model_feasibility':{
            'method':'difference_between_two_bicycle_rollouts_on_recorded_speed_trace',
            'not_a_dynamics_or_collision_proof':True,
            'steering_perturbation_amplitude_normalized':PERTURB_AMPLITUDE,
            'steering_perturbation_ramp_s':PERTURB_RAMP_S,
            'max_abs_feasible_model_steer':max(abs(r['feasible_model_steer']) for r in targets),
            'final_lateral_difference_from_empirical_m':final['perturbation_lateral_m'],
            'max_abs_lateral_difference_from_empirical_m':max(abs(r['perturbation_lateral_m']) for r in targets),
            'max_abs_heading_difference_deg':max(abs(r['perturbation_heading_deg']) for r in targets),
            'max_abs_reference_lateral_m':max(abs(r['reference_lateral_m']) for r in targets),
            'max_abs_reference_heading_deg':max(abs(r['reference_heading_change_deg']) for r in targets),
        },
        'samples':targets,
        'limitations':[
            'Not CommonRoad/Rulebook/Scenic; synthetic perturbation to empirical CARLA path',
            'Uses recorded forward speed as external input rather than predicting throttle/brake dynamics',
            'Bicycle delta is kinematically screened, not dynamically or continuously collision certified',
            'Reference steering is hypothetical and is NOT the feedforward command in either experiment arm',
            'Measurement of physical wheel steering is not frame-locked; gearbox tested only in manual first',
        ]}


def load_reference(path,obs,trial,model_hash,model):
    result=json.loads(Path(path).read_text(encoding='utf-8'))
    if (result.get('schema_version')!=SCHEMA or
        result.get('status')!='offline_reference_generated' or
        result.get('observation_sha256')!=obs['sha256'] or
        result.get('model_sha256')!=model_hash or
        result.get('positive_samples_sha256')!=trial['samples_sha256'] or
        result.get('positive_summary_sha256')!=trial['summary_sha256']):
        raise ValueError('Perturbed reference does not match original source/model')
    expected=generate_reference(trial,model,obs,model_hash)
    # An immutable, precisely reproducible reference. Reject manually edited controls/targets.
    if result!=expected:
        raise ValueError('Perturbed reference content does not match the model-generated target')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observation',type=Path,default=old.DEFAULT_OBS)
    p.add_argument('--positive-summary',type=Path,default=old.DEFAULT_POS_SUMMARY)
    p.add_argument('--positive-samples',type=Path,default=old.DEFAULT_POS_SAMPLES)
    p.add_argument('--model',type=Path,default=old.DEFAULT_MODEL)
    p.add_argument('--out',type=Path,default=old.HERE/'output/carla_calibration_6b4b/reference_v1.json')
    args=p.parse_args(argv)
    try:
        if args.out.exists():raise FileExistsError('Refusing to overwrite reference file')
        obs=old.b.read_stable_observation(args.observation)
        trial=read_trial(args.positive_summary,args.positive_samples,1)
        model=model_for_tracking(args.model,obs['sha256'],trial['samples_sha256'],trial['summary_sha256'])
        result=generate_reference(trial,model,obs,digest(args.model))
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print('M6B-4B OFFLINE REFERENCE: created',args.out)
        print('Lateral difference:',result['model_feasibility']['final_lateral_difference_from_empirical_m'],'m')
        print('No CARLA connection/tick/control, no Rulebook/Scenic.')
        return 0
    except (ValueError,KeyError,TypeError,OSError) as exc:
        print('M6B-4B REFERENCE FAILED SAFE:',type(exc).__name__+': '+str(exc))
        return 2

if __name__=='__main__':raise SystemExit(main())
