"""M6B-4B: guarded CARLA feedback-on/off tracking against SAME perturbed reference.

Default validates inputs OFFLINE; --inspect-server is read-only; --execute-track
--confirm-drive intentionally drives a temporary Tesla in *isolated* Town01.
--compare performs read-only analysis of two finished independent trials.
NEVER executes CommonRoad/Rulebook/Scenic trajectories or obstacle avoidance.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

# Provides CARLA lifecycle and actor telemetry operations.
import carla_calibrate_m6b3 as b
# Provides inherited tracking safety limits and command validation.
import carla_track_m6b4 as old
# Provides model-file validation and calibration-trial loading.
from carla_model_m6b4 import model_for_tracking, read_trial
# Shared timing, numeric, hashing, and angle primitives.
from carla_m6b_common import DT, digest, finite, wrap
from carla_feedback_reference_m6b4b import SCHEMA,load_reference

HERE=Path(__file__).resolve().parent
DEFAULT_REF=HERE/'output/carla_calibration_6b4b/reference_v1.json'
DEFAULT_OUT=HERE/'output/carla_calibration_6b4b/feedback_on_001'
SCHEMA_REPORT='m6b4b-feedback-trial-v1'
SCHEMA_COMPARISON='m6b4b-paired-comparison-v1'
MAX_START_ABSOLUTE_POSITION_ERROR_M=.02
MAX_START_ABSOLUTE_HEADING_ERROR_DEG=.5


def point_error(a,start,ref):
    s,lat,yaw=old.local_pose(a,start)
    er_s=s-finite('reference station',ref['reference_station_m'])
    er_y=lat-finite('reference lateral',ref['reference_lateral_m'])
    er_yaw=math.degrees(wrap(yaw-math.radians(finite('reference heading',ref['reference_heading_change_deg']))))
    return {
        'station_error_m':er_s,'lateral_error_m':er_y,
        'heading_error_deg':er_yaw,'position_error_m':math.hypot(er_s,er_y),
        'actual_station_m':s,'actual_lateral_m':lat,'actual_heading_change_deg':math.degrees(yaw),
        'reference_station_m':finite('ref station',ref['reference_station_m']),
        'reference_lateral_m':finite('ref lateral',ref['reference_lateral_m']),
        'reference_heading_change_deg':finite('ref heading',ref['reference_heading_change_deg'])}


def select_command(baseline_row,last_error,model,feedback_enabled):
    """Both arms use identical baseline throttle/brake/feedforward; only feedback changes."""
    th,steer,br,ff,corr=old.command_from_error(baseline_row,last_error,model)
    if not feedback_enabled:
        steer=ff;corr=0.0
    if not(-old.MAX_STEER_NORM-1e-10<=steer<=old.MAX_STEER_NORM+1e-10):
        raise ValueError('Steering exceeds demonstrated +/-0.08')
    return (th,steer,br,ff,corr)


def ensure_isolated_during_run(world,own_id,obs):
    """Detect newly spawned actors including props, not only vehicles/walkers."""
    actors=world.get_actors()
    vehicles=[a for a in actors.filter('vehicle.*') if int(a.id)!=int(own_id)]
    walkers=list(actors.filter('walker.pedestrian.*'))
    if vehicles or walkers:
        raise RuntimeError('New vehicle/pedestrian detected during isolated tracking')
    x,y=obs['body_center_cr_xy_m']
    screen=b.inspect_static_props(list(actors.filter('static.prop.*')),x,-y,-obs['yaw_cr_rad'])
    if not screen['clear_for_calibration']:
        raise RuntimeError('New static-prop corridor risk detected during isolated tracking')


def score_rows(rows):
    if not rows: return {}
    mse=sum(r['position_error_m']**2 for r in rows)/len(rows)
    heading_mse=sum(r['heading_error_deg']**2 for r in rows)/len(rows)
    corrections=[r['feedback_steer_correction'] for r in rows]
    return {
        'rms_position_error_m':math.sqrt(mse),
        'max_position_error_m':max(r['position_error_m'] for r in rows),
        'final_position_error_m':rows[-1]['position_error_m'],
        'rms_heading_error_deg':math.sqrt(heading_mse),
        'max_abs_heading_error_deg':max(abs(r['heading_error_deg']) for r in rows),
        'final_heading_error_deg':rows[-1]['heading_error_deg'],
        'max_abs_feedback_correction':max(abs(x) for x in corrections),
        'nonzero_feedback_samples':sum(abs(x)>1e-5 for x in corrections),
        'max_abs_steer_command':max(abs(r['command_steer_normalized']) for r in rows),
        'steering_saturation_samples':sum(abs(r['command_steer_normalized'])>=old.MAX_STEER_NORM-1e-6 for r in rows),
        'max_speed_mps':max(r['total_speed_mps'] for r in rows),
        'max_distance_m':max(r['distance_from_start_m'] for r in rows),
        'max_abs_cross_track_m':max(abs(r['cross_track_m']) for r in rows),
        'final_forward_speed_mps':rows[-1]['forward_speed_mps'],
    }


def summarize(rows,status,reason,collisions,cleanup,obs,model_path,trial,reference_path,feedback_on,pre,start_alignment,extra):
    return {
        'schema_version':SCHEMA_REPORT,'milestone':'6B-4B',
        'mode':'modified_empirical_reference_tracking',
        'status':status,'abort_reason':reason,
        'feedback_enabled':bool(feedback_on),'feedback_mode':'on' if feedback_on else 'off',
        'same_baseline_feedforward_for_both_modes':True,
        'reference_source':'model-predicted 2-3 cm perturbation of empirical 6B-3 CARLA path, NOT Rulebook',
        'rulebook_enabled':False,'scenic_enabled':False,'obstacle_spawned':False,
        'manual_first_gear':True,
        'carla_control_applied':extra.get('drive_control_commands_issued',0)>0,
        'source_observation_frame':obs['source_frame'],'observation_sha256':obs['sha256'],
        'model_sha256':digest(model_path),'perturbed_reference_sha256':digest(reference_path),
        'baseline_samples_sha256':trial['samples_sha256'],
        'baseline_summary_sha256':trial['summary_sha256'],
        'samples':len(rows),'sample_dt_s':DT,
        'collision_count':len(collisions),'collision_events':list(collisions),
        'absolute_start_alignment':start_alignment,
        'tracking_metrics':score_rows(rows),
        'limits':{
            'max_speed_mps':old.MAX_SPEED_MPS,'max_distance_m':old.MAX_DISPLACEMENT_M,
            'max_lateral_m':old.MAX_CROSS_M,'max_heading_deg':old.MAX_HEADING_M,
            'max_positional_tracking_error_m':old.MAX_TRACK_ERROR_M,
            'max_heading_tracking_error_deg':old.MAX_TRACK_HEADING_ERROR_DEG,
            'max_abs_steering_normalized':old.MAX_STEER_NORM,
            'max_abs_feedback_correction':old.MAX_STEER_CORRECTION,
            'max_physical_front_steer_deg':old.MAX_PHYSICAL_WHEEL_DEG,
            'max_start_absolute_position_error_m':MAX_START_ABSOLUTE_POSITION_ERROR_M,
            'max_start_absolute_heading_error_deg':MAX_START_ABSOLUTE_HEADING_ERROR_DEG},
        'preflight':pre,'cleanup':cleanup,
        'cleanup_ok':all(cleanup.get(k,False) for k in ('vehicle_destroy_returned_success','sensor_destroy_returned_success')),
        'settings_restored':cleanup.get('settings_restored',False),
        'additional_diagnostics':extra,
        'telemetry_limitations':[
            'Wheel steering angles sampled outside WorldSnapshot, not frame-locked',
            'Vehicle.get_control is applied-command readback, not independent gearbox telemetry',
            'Reference is model-perturbed empirical trace using recorded speed, not dynamically predicted path',
            'Relative tracking errors are accompanied by absolute initial pose alignment',
            'Preflight and bounding-box checks are not continuous collision/road guarantees',
        ]}


def execute(args,obs,trial,model,reference):
    carla,client,world,bp,wp_t,pre=b.preflight(args,obs,drive=True)
    original=world.get_settings()
    changed=False;ego=sensor=None
    rows=[];collisions=[];drive_commands=0;warmup=0;brake_attempted=False
    cleanup={'vehicle_destroy_returned_success':False,
             'sensor_destroy_returned_success':False,'settings_restored':False}
    status='aborted';reason=None;alignment={}
    refrows=reference['samples']; baseline=trial['rows']
    try:
        settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=DT
        changed=True
        world.apply_settings(settings)
        x,y=obs['body_center_cr_xy_m'];yaw_ca=-obs['yaw_cr_rad']
        if bp.has_attribute('role_name'):
            bp.set_attribute('role_name','milestone6b4b_feedback_'+args.feedback)
        preliminary=carla.Transform(carla.Location(x,-y,wp_t.location.z+.5),
                                    carla.Rotation(yaw=math.degrees(yaw_ca)))
        ego=world.try_spawn_actor(bp,preliminary)
        if ego is None:raise RuntimeError('Could not spawn owned Tesla')
        bb=ego.bounding_box.location
        ax,ay=b.actor_origin_for_center(x,-y,yaw_ca,bb)
        ego.set_transform(carla.Transform(carla.Location(ax,ay,wp_t.location.z+.5),
                                         carla.Rotation(yaw=math.degrees(yaw_ca))))
        ego.set_autopilot(False)
        ego.set_simulate_physics(True)
        sensor=world.spawn_actor(world.get_blueprint_library().find('sensor.other.collision'),
                                 carla.Transform(),attach_to=ego)
        if sensor is None:raise RuntimeError('Could not spawn collision sensor')
        sensor.listen(lambda e: collisions.append({'frame':int(e.frame),
            'other_actor_id':int(e.other_actor.id),'other_actor_type':str(e.other_actor.type_id)}))
        settled=0;warm_start=None
        for _ in range(80):
            ego.apply_control(carla.VehicleControl(throttle=0.,steer=0.,brake=.5))
            warmup+=1
            frame=b._get_tick(world,timeout_s=args.timeout)
            a=b.sample_actor(world,ego,carla,bb,expected_frame=frame)
            if warm_start is None:warm_start=a['t_sim_s']
            still=abs(a['vertical_speed_mps'])<=.05 and math.hypot(a['forward_speed_mps'],a['lateral_speed_mps'])<=.05
            settled=settled+1 if still else 0
            if collisions:raise RuntimeError('Collision during warmup')
            if a['t_sim_s']-warm_start>=.6 and settled>=10:break
        else:raise RuntimeError('Vehicle did not settle in 4 simulation seconds')
        start=b.sample_actor(world,ego,carla,bb)
        nominal_err=math.hypot(start['x_cr_m']-x,start['y_cr_m']-y)
        nominal_yaw=abs(math.degrees(wrap(start['yaw_cr_rad']-obs['yaw_cr_rad'])))
        origin=reference['reference_origin_cr']
        baseline_err=math.hypot(start['x_cr_m']-origin['x_cr_m'],
                                start['y_cr_m']-origin['y_cr_m'])
        baseline_yaw=abs(math.degrees(wrap(start['yaw_cr_rad']-origin['yaw_cr_rad'])))
        alignment={'observed_start_cr':[start['x_cr_m'],start['y_cr_m'],start['yaw_cr_rad']],
                   'reference_origin_cr':[origin['x_cr_m'],origin['y_cr_m'],origin['yaw_cr_rad']],
                   'error_to_empirical_reference_start_m':baseline_err,
                   'yaw_error_to_reference_start_deg':baseline_yaw,
                   'error_to_stable_nominal_start_m':nominal_err,
                   'yaw_error_to_stable_nominal_start_deg':nominal_yaw,
                   'threshold_position_m':MAX_START_ABSOLUTE_POSITION_ERROR_M,
                   'threshold_yaw_deg':MAX_START_ABSOLUTE_HEADING_ERROR_DEG}
        if (baseline_err>MAX_START_ABSOLUTE_POSITION_ERROR_M or
            baseline_yaw>MAX_START_ABSOLUTE_HEADING_ERROR_DEG or
            nominal_err>.20 or nominal_yaw>2. or
            abs(start['forward_speed_mps'])>.05 or
            abs(start['vertical_speed_mps'])>.05):
            raise RuntimeError('Absolute start pose or rest state differs from reference')
        pre['tracking_start_checked']=True
        def record_sample(a,j,phase,th,steer,br,ff,correction):
            err=point_error(a,start,refrows[j])
            rec=b.enrich(a,start,(phase,th,steer,br),j*DT)
            rec.update(b.applied_control_readback(ego))
            rec.update(err)
            rec['reference_t_s']=finite('ref time',refrows[j]['t_s'])
            rec['feedforward_steer_normalized']=ff
            rec['feedback_steer_correction']=correction
            rec['hypothetical_kinematic_reference_steer']=refrows[j]['feasible_model_steer']
            return rec
        rows.append(record_sample(start,0,'initial',0.,0.,0.,0.,0.))
        last_err=rows[0]
        for j in range(1,85):
            base=baseline[j]
            th,steer,br,ff,correction=select_command(base,last_err,model,args.feedback=='on')
            ego.apply_control(b.commanded_vehicle_control(carla,'manual_steering',th,steer,br))
            drive_commands+=1
            frame=b._get_tick(world,timeout_s=args.timeout)
            a=b.sample_actor(world,ego,carla,bb,expected_frame=frame)
            row=record_sample(a,j,base['phase'],th,steer,br,ff,correction)
            rows.append(row)
            if j>=2:old.check_expected_applied(row,th,steer,br)
            if collisions:raise RuntimeError('Collision sensor event')
            if j==10 and max(r['forward_speed_mps'] for r in rows)<.05 and max(r['distance_from_start_m'] for r in rows)<.02:
                raise RuntimeError('No longitudinal motion on throttle')
            if j==round(1.1/DT) and (max(r['forward_speed_mps'] for r in rows)<.05 or
                                         max(r['distance_from_start_m'] for r in rows)<.02):
                raise RuntimeError('No longitudinal motion before steering pulse')
            reasons=old.tracking_abort_reasons(row)
            if reasons:raise RuntimeError('Tracking safety limit: '+', '.join(reasons))
            if j%10==0:
                ensure_isolated_during_run(world,ego.id,obs)
                print('t=%.2fs v=%.3fm/s err=%.4fm yaw_err=%.3fdeg steer=%+.4f corr=%+.4f'%(
                    j*DT,row['forward_speed_mps'],row['position_error_m'],row['heading_error_deg'],steer,correction))
            last_err=row
        pulse=[r for r in rows if r['phase']=='steer_pulse']
        if max(abs(r['heading_from_start_deg']) for r in pulse)<.1 or max(abs(r['wheel_front_left_deg']) for r in pulse)<.25:
            raise RuntimeError('No measured steering response')
        if abs(rows[-1]['forward_speed_mps'])>.15:
            raise RuntimeError('Failed to stop at end of guarded run')
        if (max(r['position_error_m'] for r in rows)>old.MAX_SUCCESS_TRACK_ERROR_M or
            max(abs(r['heading_error_deg']) for r in rows)>old.MAX_SUCCESS_HEADING_ERROR_DEG):
            raise RuntimeError('Completed actuation but reference tracking exceeded success gates')
        if args.feedback=='on' and max(abs(r['feedback_steer_correction']) for r in rows)<.001:
            raise RuntimeError('Feedback-on trial had no meaningful corrective steering')
        status='completed'
    except KeyboardInterrupt:
        reason='Interrupted by user'
    except Exception as exc:
        reason=type(exc).__name__+': '+str(exc)
    # Cleanup path: always execute this block after the guarded operation to close resources or restore state.
    finally:
        if ego is not None:
            try:
                ego.apply_control(carla.VehicleControl(throttle=0.,steer=0.,brake=1.))
                brake_attempted=True
                if changed:b._get_tick(world,timeout_s=args.timeout)
            except Exception as exc:cleanup['brake_tick_error']=str(exc)
        if sensor is not None:
            try:
                sensor.stop();cleanup['sensor_destroy_returned_success']=bool(sensor.destroy())
            except Exception as exc:cleanup['sensor_cleanup_error']=str(exc)
        else:cleanup['sensor_destroy_returned_success']=True
        if ego is not None:
            try:cleanup['vehicle_destroy_returned_success']=bool(ego.destroy())
            except Exception as exc:cleanup['vehicle_cleanup_error']=str(exc)
        else:cleanup['vehicle_destroy_returned_success']=True
        if changed:
            try:
                world.apply_settings(original)
                now=world.get_settings()
                cleanup['settings_restored']=(bool(now.synchronous_mode)==bool(original.synchronous_mode) and
                                              now.fixed_delta_seconds==original.fixed_delta_seconds)
            except Exception as exc:cleanup['settings_restore_error']=str(exc)
        else:cleanup['settings_restored']=True
    if collisions and status=='completed':
        status='aborted';reason='Collision during cleanup tick'
    if not all(cleanup.get(k,False) for k in ('vehicle_destroy_returned_success',
                                               'sensor_destroy_returned_success','settings_restored')):
        status='cleanup_or_restore_failed'
        reason=reason or 'Owned actor cleanup / world settings restore unverified'
    result=summarize(rows,status,reason,collisions,cleanup,obs,args.model,trial,args.reference,
                     args.feedback=='on',pre,alignment,{
                         'drive_control_commands_issued':drive_commands,
                         'warmup_control_commands_issued':warmup,
                         'emergency_brake_attempted':brake_attempted,
                         'front_wheel_steering_not_snapshot_locked':True,
                         'reference_model_delta_not_a_steering_feedforward_command':True})
    report=b.save_results(args.out,rows,result)
    print('M6B-4B TRACK:',status,'feedback:',args.feedback,'reason:',reason)
    print('Collision count:',len(collisions),'cleanup:',cleanup)
    print('Output:',report)
    return 0 if status=='completed' else 2


def audited_run_rows(report_path,report):
    """Recompute the metrics from recorded samples; never trust just a summary."""
    sample_path=Path(report_path).parent/'calibration_samples.csv'
    with sample_path.open(newline='',encoding='utf-8') as f:
        source=list(csv.DictReader(f))
    if len(source)!=85 or report.get('samples')!=85:
        raise ValueError('Incomplete result CSV')
    numeric=('t_s','t_sim_s','position_error_m','heading_error_deg',
             'feedback_steer_correction','command_steer_normalized',
             'feedforward_steer_normalized','command_throttle','command_brake',
             'total_speed_mps','forward_speed_mps','distance_from_start_m',
             'cross_track_m','reference_t_s')
    rows=[]
    feedback_on=report['feedback_enabled']
    for j,raw in enumerate(source):
        if not all(k in raw for k in numeric):
            raise ValueError('Incomplete telemetry columns')
        r={k:finite(k,raw[k]) for k in numeric}
        r['frame']=int(raw['frame'])
        if abs(r['t_s']-j*DT)>1.e-4 or abs(r['reference_t_s']-j*DT)>1.e-4:
            raise ValueError('Reference/actual sample timestamps do not align')
        if j:
            if (r['frame']!=rows[-1]['frame']+1 or
                abs(r['t_sim_s']-rows[-1]['t_sim_s']-DT)>.002):
                raise ValueError('Tracking frames or timestamps not consecutive')
            phase,th,nominal,br=old.b.plan_command('manual_steering',(j-1)*DT,1)
            if (raw['phase']!=phase or
                abs(r['command_throttle']-th)>.001 or
                abs(r['command_brake']-br)>.001 or
                abs(r['feedforward_steer_normalized']-nominal)>.001):
                raise ValueError('Comparison arms have changed baseline control schedule')
            steering=max(-old.MAX_STEER_NORM,min(old.MAX_STEER_NORM,
                          nominal+r['feedback_steer_correction']))
            if abs(r['command_steer_normalized']-steering)>1e-4:
                raise ValueError('Logged steering command does not match feedforward + feedback')
            if not feedback_on and abs(r['feedback_steer_correction'])>1e-9:
                raise ValueError('Feedback-off trial contains corrective steering')
        rows.append(r)
    # Require the report to agree with independently recomputed CSV metrics.
    scored=score_rows(rows)
    for key,value in scored.items():
        claimed=finite('reported '+key,report.get('tracking_metrics',{}).get(key))
        if not math.isclose(claimed,value,rel_tol=1.e-8,abs_tol=1.e-8):
            raise ValueError('Report and CSV disagree on '+key)
    return scored


def compare_reports(on_path,off_path,out):
    on_path=Path(on_path);off_path=Path(off_path);out=Path(out)
    if on_path.resolve()==off_path.resolve():
        raise ValueError('Comparison requires two distinct trial reports')
    if out.exists():raise FileExistsError('Refusing to overwrite comparison output')
    on=json.loads(on_path.read_text(encoding='utf-8'))
    off=json.loads(off_path.read_text(encoding='utf-8'))
    for a,name,expected in ((on,'feedback-on',True),(off,'feedback-off',False)):
        if (a.get('schema_version')!=SCHEMA_REPORT or a.get('status')!='completed' or
            a.get('feedback_enabled') is not expected or a.get('collision_count')!=0 or
            a.get('cleanup_ok') is not True or a.get('settings_restored') is not True or
            a.get('samples')!=85 or a.get('rulebook_enabled') is not False or
            a.get('scenic_enabled') is not False or a.get('obstacle_spawned') is not False):
            raise ValueError(name+' report failed validation; cannot compare')
        align=a.get('absolute_start_alignment',{})
        if (finite(name+' start misalignment',align.get('error_to_empirical_reference_start_m')) >
                MAX_START_ABSOLUTE_POSITION_ERROR_M):
            raise ValueError(name+' has excessive start misalignment')
    for k in ('observation_sha256','model_sha256','perturbed_reference_sha256',
              'baseline_summary_sha256','baseline_samples_sha256','source_observation_frame',
              'manual_first_gear','sample_dt_s'):
        if on.get(k)!=off.get(k):
            raise ValueError('Trial source/conditions mismatch on '+k)
    if not (on['tracking_metrics']['nonzero_feedback_samples']>0 and
            off['tracking_metrics']['nonzero_feedback_samples']==0 and
            off['tracking_metrics']['max_abs_feedback_correction']==0):
        raise ValueError('Feedback conditions not reflected in logged corrections')
    a=audited_run_rows(on_path,on);bmet=audited_run_rows(off_path,off)
    rms_on=finite('feedback-on RMS',a['rms_position_error_m'])
    rms_off=finite('feedback-off RMS',bmet['rms_position_error_m'])
    if rms_off<.005:
        label='inconclusive: control run too close to modified target'
    elif rms_on<.9*rms_off:
        label='feedback_reduced_RMS_position_error_by_at_least_10_percent'
    else:
        label='feedback_improvement_not_demonstrated'
    result={
        'schema_version':SCHEMA_COMPARISON,'status':'comparison_complete',
        'conclusion':label,'same_perturbed_reference':True,
        'feedback_on_report_sha256':digest(on_path),
        'feedback_off_report_sha256':digest(off_path),
        'reference_sha256':on['perturbed_reference_sha256'],
        'model_sha256':on['model_sha256'],'observation_sha256':on['observation_sha256'],
        'rms_position_error_on_m':rms_on,'rms_position_error_off_m':rms_off,
        'rms_position_improvement_fraction':None if rms_off==0 else (rms_off-rms_on)/rms_off,
        'rms_heading_error_on_deg':a['rms_heading_error_deg'],
        'rms_heading_error_off_deg':bmet['rms_heading_error_deg'],
        'final_position_error_on_m':a['final_position_error_m'],
        'final_position_error_off_m':bmet['final_position_error_m'],
        'maximum_position_error_on_m':a['max_position_error_m'],
        'maximum_position_error_off_m':bmet['max_position_error_m'],
        'max_feedback_correction_on':a['max_abs_feedback_correction'],
        'steering_saturation_samples_on':a['steering_saturation_samples'],
        'steering_saturation_samples_off':bmet['steering_saturation_samples'],
        'initial_pose_alignment_on':on['absolute_start_alignment'],
        'initial_pose_alignment_off':off['absolute_start_alignment'],
        'limitations':['Single paired trial is not statistical validation',
                       'Feedback-on/off runs happen at different simulation times',
                       'No external disturbances were introduced; changed reference is the tracking challenge',
                       'Not Rulebook/Scenic and not an independently generated CommonRoad trajectory']}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observation',type=Path,default=old.DEFAULT_OBS)
    p.add_argument('--positive-summary',type=Path,default=old.DEFAULT_POS_SUMMARY)
    p.add_argument('--positive-samples',type=Path,default=old.DEFAULT_POS_SAMPLES)
    p.add_argument('--model',type=Path,default=old.DEFAULT_MODEL)
    p.add_argument('--reference',type=Path,default=DEFAULT_REF)
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--inspect-server',action='store_true')
    modes.add_argument('--execute-track',action='store_true')
    modes.add_argument('--compare',action='store_true')
    p.add_argument('--confirm-drive',action='store_true')
    p.add_argument('--feedback',choices=['on','off'])
    p.add_argument('--feedback-on-report',type=Path)
    p.add_argument('--feedback-off-report',type=Path)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=2000)
    p.add_argument('--timeout',type=float,default=10.)
    args=p.parse_args(argv)
    if args.confirm_drive != args.execute_track:
        p.error('Only --execute-track --confirm-drive together permits CARLA motion')
    if args.execute_track and args.feedback not in ('on','off'):
        p.error('Tracking requires --feedback on or --feedback off')
    if not args.execute_track and args.feedback is not None:
        p.error('--feedback only applies to --execute-track')
    if args.compare and not(args.feedback_on_report and args.feedback_off_report):
        p.error('--compare requires --feedback-on-report and --feedback-off-report')
    if not args.compare and (args.feedback_on_report or args.feedback_off_report):
        p.error('Report comparison inputs only allowed with --compare')
    if not math.isfinite(args.timeout) or args.timeout<=0:
        p.error('Timeout must be finite and positive')
    try:
        if args.compare:
            result=compare_reports(args.feedback_on_report,args.feedback_off_report,args.out)
            print('M6B-4B PAIRED COMPARISON:',result['conclusion'])
            print('Position RMSE (on/off):',result['rms_position_error_on_m'],result['rms_position_error_off_m'])
            print('No CARLA access; output:',args.out)
            return 0
        obs=b.read_stable_observation(args.observation)
        trial=read_trial(args.positive_summary,args.positive_samples,1)
        model=model_for_tracking(args.model,obs['sha256'],trial['samples_sha256'],trial['summary_sha256'])
        reference=load_reference(args.reference,obs,trial,digest(args.model),model)
        print('M6B-4B: validated empirical reference, modified model target, calibrated source hashes')
        print('M6B-4B: neither this reference nor these trials are CommonRoad/Rulebook/Scenic')
        if not(args.inspect_server or args.execute_track):
            print('OFFLINE ONLY: No CARLA connection, tick, actor or control')
            return 0
        if args.execute_track and args.out.exists():
            raise FileExistsError('Refusing to reuse an existing tracking output directory')
        if args.inspect_server:
            *_,pre=b.preflight(args,obs,drive=False)
            print('READ ONLY PREFLIGHT:',json.dumps(pre,indent=2))
            return 0
        return execute(args,obs,trial,model,reference)
    except (OSError,ValueError,RuntimeError,TypeError,KeyError,ImportError) as exc:
        print('M6B-4B FAILED SAFE:',type(exc).__name__+': '+str(exc))
        return 2

if __name__=='__main__':raise SystemExit(main())
