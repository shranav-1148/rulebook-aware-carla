"""M6B-5: new CommonRoad-converted ROUTE reference, low-speed CARLA tracking.

Default is OFFLINE. --inspect-server reads only. --execute-track --confirm-drive
is the sole mode which changes world sync/ticks and commands a temporary Tesla.
No original Rulebook planner, Scenic, obstacle actor or replanning is executed.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

# Provides CARLA preflight, sampling, control, ticking, and result persistence.
import carla_calibrate_m6b3 as b
# Supplies inherited low-level tracking limits and command behavior.
import carla_track_m6b4 as old
# Supplies paired-run metrics and periodic isolation checks.
import carla_feedback_track_m6b4b as prev
# Supplies the independently generated CommonRoad route reference.
import carla_route_reference_m6b5 as gen
# Shared timing, numeric, hashing, and angle primitives.
from carla_m6b_common import DT, finite, digest, wrap

HERE=Path(__file__).resolve().parent
DEFAULT_REF=HERE/'output/carla_calibration_6b5/route_reference_v1.json'
DEFAULT_OUT=HERE/'output/carla_calibration_6b5/route_on_001'
REPORT_SCHEMA='m6b5-commonroad-route-tracking-v1'
COMPARISON_SCHEMA='m6b5-route-paired-comparison-v1'
START_POSITION_TOLERANCE_M=.020
START_HEADING_TOLERANCE_DEG=.5
REFERENCE_TIME_ERROR_LIMIT_M=.20
REFERENCE_TIME_HEADING_LIMIT_DEG=4.


def point_error(actual,target):
    """Both poses are ABSOLUTE CommonRoad coordinates; do not rebase start errors away."""
    yaw=finite('reference yaw',target['yaw_cr_rad'])
    dx=finite('observed x',actual['x_cr_m'])-finite('reference x',target['x_cr_m'])
    dy=finite('observed y',actual['y_cr_m'])-finite('reference y',target['y_cr_m'])
    es=dx*math.cos(yaw)+dy*math.sin(yaw)
    ey=-dx*math.sin(yaw)+dy*math.cos(yaw)
    epsi=math.degrees(wrap(finite('observed yaw',actual['yaw_cr_rad'])-yaw))
    return {'station_error_m':es,'lateral_error_m':ey,
            'heading_error_deg':epsi,'position_error_m':math.hypot(dx,dy),
            'actual_station_m':finite('actual along',actual.get('along_track_m',0.)),
            'actual_lateral_m':finite('actual cross',actual.get('cross_track_m',0.)),
            'reference_station_m':finite('reference station',target['station_m']),
            'reference_speed_mps':finite('reference speed',target['reference_speed_mps']),
            'reference_x_cr_m':finite('reference x',target['x_cr_m']),
            'reference_y_cr_m':finite('reference y',target['y_cr_m']),
            'reference_yaw_cr_rad':yaw}


def command_for_reference(ref,err,model,feedback_on):
    """Both arms have the SAME route-curvature feedforward, throttle and brake."""
    row={'command_throttle':finite('throttle',ref['command_throttle']),
         'command_brake':finite('brake',ref['command_brake']),
         'command_steer_normalized':finite('route steer',ref['feedforward_steer_normalized'])}
    th,steer,br,ff,corr=old.command_from_error(row,err,model)
    if not feedback_on:steer=ff;corr=0.
    if abs(steer)>old.MAX_STEER_NORM+1e-9 or abs(corr)>old.MAX_STEER_CORRECTION+1e-9:
        raise ValueError('Unvalidated control magnitude')
    return th,steer,br,ff,corr


def evaluate_road(actual,road,rectangle):
    """Live body rectangle sampled against the previously loaded CommonRoad roadway."""
    poly=rectangle(finite('x',actual['x_cr_m']),finite('y',actual['y_cr_m']),
                   finite('yaw',actual['yaw_cr_rad']),4.791779518127441,2.163450002670288)
    area=poly.difference(road).area
    if not math.isfinite(area) or area>1e-5:
        raise RuntimeError('Measured car footprint outside validated CommonRoad road area')
    return area


def make_report(rows,status,reason,collisions,cleanup,obs,args,ref,pre,alignment,extra):
    return {'schema_version':REPORT_SCHEMA,'milestone':'6B-5',
        'mode':'independent_CommonRoad_route_tracking','status':status,'abort_reason':reason,
        'feedback_enabled':args.feedback=='on','feedback_mode':args.feedback,
        'commonroad_route_source':True,'original_rulebook_planner_used':False,
        'scenic_enabled':False,'obstacle_spawned':False,'manual_first_gear':True,
        'carla_control_applied':extra['drive_commands']>0,
        'reference_sha256':digest(args.reference),'route_sha256':ref['route_sha256'],
        'map_sha256':ref['map_sha256'],'model_sha256':ref['calibrated_model_sha256'],
        'observation_sha256':obs['sha256'],'source_observation_frame':obs['source_frame'],
        'reference_type':ref['reference_source'],
        'samples':len(rows),'sample_dt_s':DT,
        'tracking_metrics':prev.score_rows(rows),
        'maximum_sampled_live_road_outside_area_m2':extra.get('max_road_outside_area_m2'),
        'collision_count':len(collisions),'collision_events':list(collisions),
        'start_alignment':alignment,'preflight':pre,'cleanup':cleanup,
        'cleanup_ok':all(cleanup.get(x,False) for x in ('vehicle_destroy_returned_success','sensor_destroy_returned_success')),
        'settings_restored':cleanup.get('settings_restored',False),
        'limits':{'speed_mps':old.MAX_SPEED_MPS,'displacement_m':old.MAX_DISPLACEMENT_M,
                  'lateral_m':old.MAX_CROSS_M,'heading_change_deg':old.MAX_HEADING_M,
                  'tracking_position_error_m':REFERENCE_TIME_ERROR_LIMIT_M,
                  'tracking_heading_error_deg':REFERENCE_TIME_HEADING_LIMIT_DEG,
                  'normalized_steering':old.MAX_STEER_NORM,
                  'normalized_feedback_correction':old.MAX_STEER_CORRECTION,
                  'measured_front_wheel_angle_deg':old.MAX_PHYSICAL_WHEEL_DEG,
                  'start_position_error_m':START_POSITION_TOLERANCE_M,
                  'start_heading_error_deg':START_HEADING_TOLERANCE_DEG},
        'run_diagnostics':extra,
        'limitations':[
            'Reference comes from CommonRoad-converted Town01 route, not the original Rulebook algorithm',
            'Nominal speed profile is not fitted vehicle acceleration/braking dynamics',
            'No obstacle, Scenic or autonomous replanning',
            'Collision sensor, static-prop checks and sampled road polygon are not continuous safety certification',
            'Observed steering angles are queried after world snapshot, not frame-locked',
            'This is a single bounded low-speed CARLA run; no statistical guarantee']}


def execute(args,obs,model,ref,geometry):
    # No CARLA access before reference/provenance/road checks have completed.
    route,road,lanelets,pose_at,project,rectangle=geometry
    carla,client,world,bp,wp_t,pre=b.preflight(args,obs,drive=True)
    original=world.get_settings();changed=False;ego=sensor=None
    rows=[];collisions=[];commands=0;warmup=0;brake_attempted=False
    cleanup={'vehicle_destroy_returned_success':False,'sensor_destroy_returned_success':False,'settings_restored':False}
    alignment={};max_outside=0.;status='aborted';reason=None
    targets=ref['samples']
    try:
        settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=DT
        changed=True;world.apply_settings(settings)
        x,y=obs['body_center_cr_xy_m'];yaw_ca=-obs['yaw_cr_rad']
        if bp.has_attribute('role_name'):
            bp.set_attribute('role_name','m6b5_route_tracking_'+args.feedback)
        initial=carla.Transform(carla.Location(x,-y,wp_t.location.z+.5),
                                carla.Rotation(yaw=math.degrees(yaw_ca)))
        ego=world.try_spawn_actor(bp,initial)
        if ego is None:raise RuntimeError('Could not spawn owned Tesla')
        bb=ego.bounding_box.location
        ax,ay=b.actor_origin_for_center(x,-y,yaw_ca,bb)
        ego.set_transform(carla.Transform(carla.Location(ax,ay,wp_t.location.z+.5),
                                         carla.Rotation(yaw=math.degrees(yaw_ca))))
        ego.set_autopilot(False);ego.set_simulate_physics(True)
        sensor=world.spawn_actor(world.get_blueprint_library().find('sensor.other.collision'),
                                 carla.Transform(),attach_to=ego)
        if sensor is None:raise RuntimeError('Could not spawn collision sensor')
        sensor.listen(lambda event:collisions.append({'frame':int(event.frame),
            'other_actor_id':int(event.other_actor.id),'other_actor_type':str(event.other_actor.type_id)}))
        settled=0;warm_start=None
        for i in range(80):
            ego.apply_control(carla.VehicleControl(throttle=0.,steer=0.,brake=.5))
            warmup+=1
            f=b._get_tick(world,timeout_s=args.timeout)
            a=b.sample_actor(world,ego,carla,bb,expected_frame=f)
            if warm_start is None:warm_start=a['t_sim_s']
            still=abs(a['vertical_speed_mps'])<=.05 and math.hypot(a['forward_speed_mps'],a['lateral_speed_mps'])<=.05
            settled=settled+1 if still else 0
            if collisions:raise RuntimeError('Collision during warmup')
            if a['t_sim_s']-warm_start>=.6 and settled>=10:break
        else:raise RuntimeError('Vehicle did not settle in four simulation seconds')
        start=b.sample_actor(world,ego,carla,bb)
        target0=targets[0]
        dp=math.hypot(start['x_cr_m']-target0['x_cr_m'],start['y_cr_m']-target0['y_cr_m'])
        dh=abs(math.degrees(wrap(start['yaw_cr_rad']-target0['yaw_cr_rad'])))
        alignment={'start_position_error_m':dp,'start_heading_error_deg':dh,
                   'observed_start_cr':[start['x_cr_m'],start['y_cr_m'],start['yaw_cr_rad']],
                   'route_reference_start_cr':[target0['x_cr_m'],target0['y_cr_m'],target0['yaw_cr_rad']]}
        if (dp>START_POSITION_TOLERANCE_M or dh>START_HEADING_TOLERANCE_DEG or
            abs(start['forward_speed_mps'])>.05 or abs(start['vertical_speed_mps'])>.05):
            raise RuntimeError('Actual start pose/rest condition does not match CommonRoad route reference')
        max_outside=max(max_outside,evaluate_road(start,road,rectangle))
        pre['tracking_start_checked']=True
        def record(a,i,th,steer,br,ff,corr):
            target=targets[i]
            r=b.enrich(a,start,(target['phase'],th,steer,br),i*DT)
            r.update(b.applied_control_readback(ego))
            r.update(point_error(a,target))
            r['reference_t_s']=finite('time',target['t_s'])
            r['feedforward_steer_normalized']=ff
            r['feedback_steer_correction']=corr
            r['reference_acceleration_mps2']=target['reference_acceleration_mps2']
            return r
        rows.append(record(start,0,0.,0.,0.,0.,0.))
        last_error=rows[-1]
        for j in range(1,gen.COUNT):
            target=targets[j]
            th,steer,br,ff,corr=command_for_reference(target,last_error,model,args.feedback=='on')
            ego.apply_control(b.commanded_vehicle_control(carla,'manual_steering',th,steer,br))
            commands+=1
            frame=b._get_tick(world,timeout_s=args.timeout)
            a=b.sample_actor(world,ego,carla,bb,expected_frame=frame)
            r=record(a,j,th,steer,br,ff,corr)
            rows.append(r)
            if j>=2:old.check_expected_applied(r,th,steer,br)
            if collisions:raise RuntimeError('Collision sensor event')
            max_outside=max(max_outside,evaluate_road(a,road,rectangle))
            if j==10 and max(q['forward_speed_mps'] for q in rows)<.05:
                raise RuntimeError('No measurable longitudinal motion in first half-second')
            bad=old.tracking_abort_reasons(r)
            if bad:raise RuntimeError('Tracking motion safety limit: '+', '.join(bad))
            if (r['position_error_m']>REFERENCE_TIME_ERROR_LIMIT_M or
                abs(r['heading_error_deg'])>REFERENCE_TIME_HEADING_LIMIT_DEG):
                raise RuntimeError('New route time-aligned tracking-error safety limit')
            if j%10==0:
                prev.ensure_isolated_during_run(world,ego.id,obs)
                print('t=%.2fs speed=%.3fm/s route_err=%.4fm yaw_err=%.3fdeg steer=%+.4f'%(
                    j*DT,r['forward_speed_mps'],r['position_error_m'],r['heading_error_deg'],steer))
            last_error=r
        if abs(rows[-1]['forward_speed_mps'])>.15:
            raise RuntimeError('Vehicle failed to stop at end of route reference')
        if (max(r['position_error_m'] for r in rows)>old.MAX_SUCCESS_TRACK_ERROR_M or
            max(abs(r['heading_error_deg']) for r in rows)>old.MAX_SUCCESS_HEADING_ERROR_DEG):
            raise RuntimeError('Completed actuation but reference tracking exceeded strict completion gate')
        # No minimum correction magnitude: a straight route may need no intervention.
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
            try:sensor.stop();cleanup['sensor_destroy_returned_success']=bool(sensor.destroy())
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
    if collisions and status=='completed':status='aborted';reason='Late collision during braking/cleanup'
    if not all(cleanup.get(k,False) for k in cleanup if k.endswith('_returned_success') or k=='settings_restored'):
        status='cleanup_or_restore_failed';reason=reason or 'Actor cleanup/world setting restoration unverified'
    result=make_report(rows,status,reason,collisions,cleanup,obs,args,ref,pre,alignment,{
        'drive_commands':commands,'warmup_commands':warmup,'emergency_brake_attempted':brake_attempted,
        'max_road_outside_area_m2':max_outside,
        'reference_speed_is_model_hypothesis':True,'automatic_gearbox_tested':False})
    output=b.save_results(args.out,rows,result)
    print('M6B-5 ROUTE TRACK:',status,'feedback:',args.feedback,'reason:',reason)
    print('Collisions:',len(collisions),'cleanup:',cleanup,'output:',output)
    return 0 if status=='completed' else 2


def audited_rows(report_path,report,reference):
    """Independently recompute pose errors and command consistency from CSV."""
    path=Path(report_path).parent/'calibration_samples.csv'
    with path.open(newline='',encoding='utf-8') as f:raw=list(csv.DictReader(f))
    if len(raw)!=gen.COUNT or report['samples']!=gen.COUNT:
        raise ValueError('Missing/incomplete route tracking sample CSV')
    rows=[]
    for j,row in enumerate(raw):
        target=reference['samples'][j]
        t=finite('time',row['t_s']);rt=finite('ref time',row['reference_t_s'])
        if abs(t-j*DT)>1e-4 or abs(rt-t)>1e-4:
            raise ValueError('Route reference/actual timestamp mismatch')
        if j:
            if int(row['frame'])!=int(raw[j-1]['frame'])+1 or abs(finite('sim elapsed',row['t_sim_s'])-finite('prev sim elapsed',raw[j-1]['t_sim_s'])-DT)>.002:
                raise ValueError('Nonconsecutive simulation frames/timing')
            for fld,required in (('command_throttle',target['command_throttle']),
                                 ('command_brake',target['command_brake']),
                                 ('feedforward_steer_normalized',target['feedforward_steer_normalized'])):
                if abs(finite(fld,row[fld])-finite('required',required))>1e-4:
                    raise ValueError('Changed reference feedforward or longitudinal control')
            corr=finite('correction',row['feedback_steer_correction'])
            if abs(corr)>old.MAX_STEER_CORRECTION+1e-6:raise ValueError('Correction limit exceeded')
            commanded=max(-old.MAX_STEER_NORM,min(old.MAX_STEER_NORM,
                target['feedforward_steer_normalized']+corr))
            if abs(finite('steer',row['command_steer_normalized'])-commanded)>1e-4:
                raise ValueError('Steering command not equal to feedforward plus correction')
            if not report['feedback_enabled'] and abs(corr)>1e-9:
                raise ValueError('Feedback-off has nonzero corrective commands')
        actual={k:finite(k,row[k]) for k in ('x_cr_m','y_cr_m','yaw_cr_rad')}
        computed=point_error(actual,target)
        for k in ('station_error_m','lateral_error_m','position_error_m','heading_error_deg'):
            if abs(finite('logged '+k,row[k])-computed[k])>1e-6:
                raise ValueError('Logged route error differs from independently recomputed absolute pose error: '+k)
        values={k:finite(k,row[k]) for k in ('position_error_m','heading_error_deg',
             'feedback_steer_correction','command_steer_normalized','total_speed_mps',
             'distance_from_start_m','cross_track_m','forward_speed_mps')}
        rows.append(values)
    summary=prev.score_rows(rows)
    for k,v in summary.items():
        if abs(finite('reported metric '+k,report['tracking_metrics'][k])-v)>1e-7:
            raise ValueError('Mismatch between report and CSV: '+k)
    return summary


def compare(on_path,off_path,out,reference):
    on_path=Path(on_path);off_path=Path(off_path);out=Path(out)
    if on_path.resolve()==off_path.resolve() or out.exists():
        raise ValueError('Distinct reports and new comparison output required')
    on=json.loads(on_path.read_text(encoding='utf-8'))
    off=json.loads(off_path.read_text(encoding='utf-8'))
    for label,r,enabled in (('ON',on,True),('OFF',off,False)):
        if (r.get('schema_version')!=REPORT_SCHEMA or r.get('status')!='completed' or
            r.get('feedback_enabled') is not enabled or r.get('collision_count')!=0 or
            r.get('cleanup_ok') is not True or r.get('settings_restored') is not True or
            r.get('samples')!=gen.COUNT or r.get('obstacle_spawned') is not False or
            r.get('original_rulebook_planner_used') is not False):
            raise ValueError('Invalid '+label+' tracking report')
        if (r.get('reference_sha256')!=digest_reference_source(reference) or
            finite('start error',r['start_alignment']['start_position_error_m'])>START_POSITION_TOLERANCE_M):
            raise ValueError('Reference hash/start alignment mismatch in '+label)
    for k in ('reference_sha256','route_sha256','map_sha256','model_sha256','observation_sha256',
              'source_observation_frame','sample_dt_s','manual_first_gear'):
        if on.get(k)!=off.get(k):raise ValueError('Paired report provenance mismatch: '+k)
    a=audited_rows(on_path,on,reference);z=audited_rows(off_path,off,reference)
    if a['nonzero_feedback_samples']==0 and a['max_abs_feedback_correction']!=0:
        raise ValueError('Feedback metrics inconsistent')
    rms_on=a['rms_position_error_m'];rms_off=z['rms_position_error_m']
    if rms_off<.005:conclusion='inconclusive_small_off_error'
    elif rms_on<.90*rms_off:conclusion='feedback_reduced_RMS_error_at_least_10_percent'
    else:conclusion='feedback_improvement_not_demonstrated'
    result={'schema_version':COMPARISON_SCHEMA,'status':'comparison_complete','conclusion':conclusion,
        'same_independent_commonroad_route_reference':True,
        'feedback_on_sha256':digest(on_path),'feedback_off_sha256':digest(off_path),
        'reference_sha256':on['reference_sha256'],'route_sha256':on['route_sha256'],
        'rms_position_on_m':rms_on,'rms_position_off_m':rms_off,
        'rms_position_improvement_fraction':None if rms_off==0 else (rms_off-rms_on)/rms_off,
        'rms_heading_on_deg':a['rms_heading_error_deg'],'rms_heading_off_deg':z['rms_heading_error_deg'],
        'final_position_on_m':a['final_position_error_m'],'final_position_off_m':z['final_position_error_m'],
        'max_position_on_m':a['max_position_error_m'],'max_position_off_m':z['max_position_error_m'],
        'feedback_on_nonzero_samples':a['nonzero_feedback_samples'],
        'limitations':['Not the Rulebook planner; path follows existing CommonRoad-converted route',
                       'Single pair is not a statistical result',
                       'Nominal reference speed is a model hypothesis, not calibrated longitudinal dynamics']}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return result


def digest_reference_source(reference):
    # The reference object needs its actual source path to check exact bytes.
    return digest(reference['_loaded_from'])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observation',type=Path,default=gen.DEFAULT_OBS)
    p.add_argument('--model',type=Path,default=gen.DEFAULT_MODEL)
    p.add_argument('--route',type=Path,default=gen.DEFAULT_ROUTE)
    p.add_argument('--map',type=Path,default=gen.DEFAULT_MAP)
    p.add_argument('--source-summary','--4eb-summary',dest='source_summary',type=Path,default=gen.DEFAULT_SUMMARY)
    p.add_argument('--reference',type=Path,default=DEFAULT_REF)
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--inspect-server',action='store_true')
    mode.add_argument('--execute-track',action='store_true')
    mode.add_argument('--compare',action='store_true')
    p.add_argument('--confirm-drive',action='store_true')
    p.add_argument('--feedback',choices=['on','off'])
    p.add_argument('--feedback-on-report',type=Path)
    p.add_argument('--feedback-off-report',type=Path)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=2000)
    p.add_argument('--timeout',type=float,default=10.)
    a=p.parse_args(argv)
    if a.confirm_drive != a.execute_track:p.error('CARLA movement needs --execute-track AND --confirm-drive')
    if a.execute_track and a.feedback not in ('on','off'):p.error('Select --feedback on or off')
    if not a.execute_track and a.feedback is not None:p.error('--feedback applies only to driving mode')
    if a.compare!=(a.feedback_on_report is not None and a.feedback_off_report is not None):
        p.error('--compare requires two --feedback-*-report files, and they only apply to --compare')
    if not math.isfinite(a.timeout) or a.timeout<=0:p.error('Timeout must be positive and finite')
    try:
        obs=b.read_stable_observation(a.observation)
        model=gen.validated_model(a.model,obs['sha256'])
        geometry=gen.geometry_from_project(a.route,a.map)
        reference=gen.load_reference(a.reference,obs,model,a.model,a.route,a.map,a.source_summary,geometry)
        reference['_loaded_from']=str(a.reference)
        print('M6B-5: verified independent CommonRoad-converted route/map/reference content and source hashes')
        if a.compare:
            result=compare(a.feedback_on_report,a.feedback_off_report,a.out,reference)
            print('OFFLINE PAIRED COMPARISON:',result['conclusion'],'output:',a.out);return 0
        if a.execute_track and a.out.exists():raise FileExistsError('Refusing to overwrite previous CARLA run directory')
        if a.inspect_server:
            *_,pre=b.preflight(a,obs,drive=False)
            print('READ ONLY PREFLIGHT:',json.dumps(pre,indent=2));return 0
        if not a.execute_track:
            print('OFFLINE ONLY: No CARLA connection/tick/control. Route reference valid.');return 0
        return execute(a,obs,model,reference,geometry)
    except (OSError,ValueError,RuntimeError,TypeError,KeyError,ImportError) as exc:
        print('M6B-5 FAILED SAFE:',type(exc).__name__+': '+str(exc));return 2

if __name__=='__main__':raise SystemExit(main())
