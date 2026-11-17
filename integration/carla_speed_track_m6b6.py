"""M6B-6: isolated low-speed CARLA speed-feedback test on the SAME 6B-5 CommonRoad reference.

Offline validation is default. --inspect-server reads only. Only --execute-track
--confirm-drive deliberately drives, with the inherited 6B-5 road/collision safeguards.
Steering feedforward is IDENTICAL in both arms and steering feedback is disabled.
This is NOT upstream Rulebook or a longitudinal dynamics certification.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import carla_calibrate_m6b3 as b
import carla_track_m6b4 as old
import carla_feedback_track_m6b4b as prev
import carla_route_reference_m6b5 as gen
import carla_route_track_m6b5 as core
import carla_speed_fit_m6b6 as fit
from carla_model_m6b4 import DT,finite,digest,wrap

HERE=Path(__file__).resolve().parent
DEFAULT_REF=HERE/'output/carla_calibration_6b5/route_reference_v1.json'
DEFAULT_OUT=HERE/'output/carla_calibration_6b6/speed_on_001'
DEFAULT_FIT=fit.DEFAULT_OUT
REPORT_SCHEMA='m6b6-longitudinal-tracking-v1'
COMPARISON_SCHEMA='m6b6-speed-paired-comparison-v1'
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


# Conservative command bounds: no higher throttle/brake than previously tested.
MAX_THROTTLE=.25
MAX_BRAKE=.65
MAX_THROTTLE_CORRECTION=.10
MAX_BRAKE_CORRECTION=.25
SPEED_KP_THROTTLE=.20  # normalized throttle / (m/s speed deficit)
SPEED_KP_BRAKE=.18     # normalized brake / (m/s speed excess)


def command_for_reference(ref, last, model, fit_doc, feedback_on):
    """Compare speed feedback only, holding lateral control IDENTICAL across arms.

    The same route curvature steering feedforward and nominal throttle/brake
    schedule are used in both modes. Corrections depend on previous measured
    speed, never a future CARLA observation. No simultaneous throttle/brake.
    """
    raw=core_command_for_reference(ref, last, model)
    th, steer, br, ff=raw
    speed_ref=finite('reference speed',ref['reference_speed_mps'])
    speed_now=finite('previous measured speed',last.get('forward_speed_mps',0.))
    error=speed_ref-speed_now
    if not -0.5<=speed_now<=old.MAX_SPEED_MPS+1e-6:
        raise RuntimeError('Unsafe observed speed before speed feedback')
    if not feedback_on:
        return th,steer,br,ff,0.,0.,error
    if (fit_doc.get('status')!='descriptive_offline_fit' or
        fit_doc.get('limits',{}).get('manual_first_gear_only') is not True):
        raise ValueError('Invalid longitudinal response source')
    # Guard at zero-reference speed: no throttle after planned stop begins.
    if th>0:
        target=th+SPEED_KP_THROTTLE*error
        chosen_th=max(0.,min(MAX_THROTTLE,max(th-MAX_THROTTLE_CORRECTION,
                   min(th+MAX_THROTTLE_CORRECTION,target))))
        return chosen_th,steer,0.,ff,chosen_th-th,0.,error
    # In coast and brake phases, throttle stays 0: the new controller NEVER
    # accelerates into the prescribed stopping region.
    if br>0:
        # Baseline braking .65 is extremely strong in the observed samples.
        # Allow only modest, bounded reduction if car is already below target.
        chosen_br=max(0.,min(MAX_BRAKE,br+max(-MAX_BRAKE_CORRECTION,
                         min(MAX_BRAKE_CORRECTION,-SPEED_KP_BRAKE*error))))
    else:
        # 6B-6 coast-phase safety correction (trial speed_on_001):
        # CARLA accepted 0.001825 normalized brake at t=1.90 s and speed
        # dropped by 1.047 m/s in one 0.05 s tick. The plant's response
        # to near-zero brake is NOT calibrated. Preserve baseline coasting;
        # do not issue feedback brake (or throttle) in this phase.
        chosen_br=0.
    return 0.,steer,chosen_br,ff,0.,chosen_br-br,error


def core_command_for_reference(ref,last,model):
    """Reuses 6B-5 route curvature feedforward with NO steering feedback."""
    th,steer,br,ff,corr=core.command_for_reference(ref,last,model,False)
    assert corr==0.
    return th,steer,br,ff


def evaluate_road(actual,road,rectangle):
    """Live body rectangle sampled against the previously loaded CommonRoad roadway."""
    poly=rectangle(finite('x',actual['x_cr_m']),finite('y',actual['y_cr_m']),
                   finite('yaw',actual['yaw_cr_rad']),4.791779518127441,2.163450002670288)
    area=poly.difference(road).area
    if not math.isfinite(area) or area>1e-5:
        raise RuntimeError('Measured car footprint outside validated CommonRoad road area')
    return area


def speed_metrics(rows):
    if not rows:return {'count':0}
    errors=[finite('speed error',r['speed_error_mps']) for r in rows]
    return {'count':len(rows),
        'rms_speed_error_mps':math.sqrt(sum(e*e for e in errors)/len(errors)),
        'mean_absolute_speed_error_mps':sum(abs(e) for e in errors)/len(errors),
        'max_absolute_speed_error_mps':max(abs(e) for e in errors),
        'final_speed_error_mps':errors[-1],
        'nonzero_throttle_correction_samples':sum(abs(r['throttle_correction'])>1e-7 for r in rows),
        'nonzero_brake_correction_samples':sum(abs(r['brake_correction'])>1e-7 for r in rows)}


def make_report(rows,status,reason,collisions,cleanup,obs,args,ref,pre,alignment,extra):
    return {'schema_version':REPORT_SCHEMA,'milestone':'6B-6',
        'mode':'independent_commonroad_route_speed_tracking','status':status,'abort_reason':reason,
        'speed_feedback_enabled':args.speed_feedback=='on','speed_feedback_mode':args.speed_feedback,
        'steering_feedback_enabled':False,'same_route_feedforward_and_nominal_schedule':True,
        'original_rulebook_planner_used':False,'scenic_enabled':False,'obstacle_spawned':False,
        'manual_first_gear':True,'carla_control_applied':extra['drive_commands']>0,
        'reference_sha256':digest(args.reference),'longitudinal_fit_sha256':digest(args.longitudinal_fit),
        'route_sha256':ref['route_sha256'],'map_sha256':ref['map_sha256'],
        'model_sha256':ref['calibrated_model_sha256'],
        'observation_sha256':obs['sha256'],'source_observation_frame':obs['source_frame'],
        'samples':len(rows),'sample_dt_s':DT,
        'speed_metrics':speed_metrics(rows),
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
            'throttle_max':MAX_THROTTLE,'brake_max':MAX_BRAKE,
            'throttle_correction_max':MAX_THROTTLE_CORRECTION,
            'brake_correction_max':MAX_BRAKE_CORRECTION,
            'normalized_steering':old.MAX_STEER_NORM,
            'measured_front_wheel_angle_deg':old.MAX_PHYSICAL_WHEEL_DEG,
            'start_position_error_m':START_POSITION_TOLERANCE_M,
            'start_heading_error_deg':START_HEADING_TOLERANCE_DEG},
        'run_diagnostics':extra,
        'limitations':[
            'Same independent CommonRoad-converted Town01 route as 6B-5; NOT original Rulebook',
            'Only steering feedforward; no steering feedback in either comparison arm',
            'Speed reference is an UNFITTED hypothesis with very sharp nominal braking',
            'Empirical longitudinal source is sparse especially during strong braking',
            'A single comparison cannot establish a calibrated dynamics model or statistical performance',
            'Actuator readback and wheel-angle queries are not snapshot frame-locked',
            'Sampled road/collision checks are not continuous safety guarantees']}


def execute(args,obs,model,ref,geometry,fit_doc):
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
            bp.set_attribute('role_name','m6b6_speed_tracking_'+args.speed_feedback)
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
            r['speed_error_mps']=finite('actual speed',a['forward_speed_mps'])-finite('ref speed',target['reference_speed_mps'])
            r['baseline_throttle']=target['command_throttle']
            r['baseline_brake']=target['command_brake']
            r['throttle_correction']=th-target['command_throttle']
            r['brake_correction']=br-target['command_brake']
            return r
        rows.append(record(start,0,0.,0.,0.,0.,0.))
        last_error=rows[-1]
        for j in range(1,gen.COUNT):
            target=targets[j]
            th,steer,br,ff,corr_th,corr_br,_=command_for_reference(target,last_error,model,fit_doc,args.speed_feedback=='on')
            corr=0.
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
    print('M6B-6 SPEED TRACK:',status,'speed-feedback:',args.speed_feedback,'reason:',reason)
    print('Collisions:',len(collisions),'cleanup:',cleanup,'output:',output)
    return 0 if status=='completed' else 2


def audited_rows(report_path,report,reference):
    """Validate time alignment, feedforward, recorded speed errors and commands."""
    path=Path(report_path).parent/'calibration_samples.csv'
    with path.open(newline='',encoding='utf-8') as f:raw=list(csv.DictReader(f))
    if len(raw)!=gen.COUNT or report.get('samples')!=gen.COUNT:
        raise ValueError('Missing/incomplete speed-tracking samples')
    rows=[]
    for j,r in enumerate(raw):
        target=reference['samples'][j]
        if abs(finite('t_s',r['t_s'])-j*DT)>1e-4 or abs(finite('ref t',r['reference_t_s'])-j*DT)>1e-4:
            raise ValueError('Actual/reference timestamps differ')
        if j:
            p=raw[j-1]
            if (int(r['frame'])!=int(p['frame'])+1 or
                abs(finite('sim',r['t_sim_s'])-finite('prev sim',p['t_sim_s'])-DT)>.002):
                raise ValueError('Frame or simulation time discontinuity')
            th,br=finite('throttle',r['command_throttle']),finite('brake',r['command_brake'])
            baseline_th=finite('nominal throttle',target['command_throttle'])
            baseline_br=finite('nominal brake',target['command_brake'])
            if (not 0<=th<=MAX_THROTTLE+1e-6 or not 0<=br<=MAX_BRAKE+1e-6 or th*br>1e-9):
                raise ValueError('Unsupported throttle/brake or overlapping pedals')
            if (abs(finite('ff steer',r['feedforward_steer_normalized'])-target['feedforward_steer_normalized'])>1e-5 or
                abs(finite('command steer',r['command_steer_normalized'])-target['feedforward_steer_normalized'])>1e-5 or
                abs(finite('steering correction',r['feedback_steer_correction']))>1e-9):
                raise ValueError('Steering feedback or differing feedforward in comparison')
            if (abs(finite('throttle correction',r['throttle_correction'])-(th-baseline_th))>1e-6 or
                abs(finite('brake correction',r['brake_correction'])-(br-baseline_br))>1e-6):
                raise ValueError('Misreported command correction')
            if not report['speed_feedback_enabled'] and (abs(th-baseline_th)>1e-9 or abs(br-baseline_br)>1e-9):
                raise ValueError('Speed-feedback-OFF altered nominal command schedule')
            if abs(th-baseline_th)>MAX_THROTTLE_CORRECTION+1e-6 or abs(br-baseline_br)>MAX_BRAKE_CORRECTION+1e-6:
                raise ValueError('Speed feedback correction exceeds bound')
            if (abs(finite('applied throttle',r['applied_throttle'])-th)>.02 or
                abs(finite('applied brake',r['applied_brake'])-br)>.05 or
                str(r['applied_manual_gear_shift'])!='True' or
                int(r['applied_gear'])!=1 or str(r['applied_hand_brake'])!='False' or
                str(r['applied_reverse'])!='False'):
                raise ValueError('Control readback mismatch')
        actual={k:finite(k,r[k]) for k in ('x_cr_m','y_cr_m','yaw_cr_rad')}
        calc=core.point_error(actual,target)
        for k in ('station_error_m','lateral_error_m','position_error_m','heading_error_deg'):
            if abs(finite('logged '+k,r[k])-calc[k])>1e-6:
                raise ValueError('Logged trajectory error does not match independent pose error')
        ve=finite('v measured',r['forward_speed_mps'])-finite('v ref',target['reference_speed_mps'])
        if abs(finite('logged speed error',r['speed_error_mps'])-ve)>1e-6:
            raise ValueError('Logged speed error differs from measured minus reference')
        rows.append({k:finite(k,r[k]) for k in ('position_error_m','heading_error_deg',
            'feedback_steer_correction','command_steer_normalized','total_speed_mps',
            'distance_from_start_m','cross_track_m','forward_speed_mps',
            'speed_error_mps','throttle_correction','brake_correction')})
    v=speed_metrics(rows)
    pos=prev.score_rows(rows)
    for metrics,reported in ((v,report['speed_metrics']),(pos,report['tracking_metrics'])):
        for k,x in metrics.items():
            if abs(finite('reported metric '+k,reported[k])-x)>1e-7:
                raise ValueError('Summary/CSV metric discrepancy: '+k)
    return v,pos


def compare(on_path,off_path,out,reference):
    on_path=Path(on_path);off_path=Path(off_path);out=Path(out)
    if on_path.resolve()==off_path.resolve() or out.exists():
        raise ValueError('Comparison requires distinct source reports and new output')
    on=json.loads(on_path.read_text(encoding='utf-8'))
    off=json.loads(off_path.read_text(encoding='utf-8'))
    for tag,rep,enabled in (('ON',on,True),('OFF',off,False)):
        if (rep.get('schema_version')!=REPORT_SCHEMA or rep.get('status')!='completed' or
            rep.get('speed_feedback_enabled') is not enabled or rep.get('steering_feedback_enabled') is not False or
            rep.get('collision_count')!=0 or rep.get('cleanup_ok') is not True or
            rep.get('settings_restored') is not True or rep.get('samples')!=gen.COUNT or
            rep.get('original_rulebook_planner_used') is not False or rep.get('obstacle_spawned') is not False or
            rep.get('reference_sha256')!=digest(reference['_loaded_from']) or
            finite('start alignment',rep['start_alignment']['start_position_error_m'])>START_POSITION_TOLERANCE_M):
            raise ValueError('Invalid '+tag+' completed speed trial')
    for k in ('reference_sha256','route_sha256','map_sha256','model_sha256',
              'observation_sha256','longitudinal_fit_sha256','source_observation_frame',
              'sample_dt_s','manual_first_gear'):
        if on.get(k)!=off.get(k):raise ValueError('Paired provenance mismatch: '+k)
    metrics_on,pos_on=audited_rows(on_path,on,reference)
    metrics_off,pos_off=audited_rows(off_path,off,reference)
    on_rms=metrics_on['rms_speed_error_mps'];off_rms=metrics_off['rms_speed_error_mps']
    if (metrics_on['nonzero_throttle_correction_samples']+metrics_on['nonzero_brake_correction_samples']==0):
        conclusion='inconclusive_no_active_speed_correction'
    elif off_rms<.01:conclusion='inconclusive_small_fixed_control_speed_error'
    elif on_rms<.90*off_rms:conclusion='speed_feedback_reduced_RMS_speed_error_at_least_10_percent'
    else:conclusion='speed_feedback_improvement_not_demonstrated'
    result={'schema_version':COMPARISON_SCHEMA,'status':'comparison_complete',
        'conclusion':conclusion,'same_commonroad_route_reference':True,
        'steering_strategy_same_feedforward_only':True,
        'feedback_on_report_sha256':digest(on_path),'feedback_off_report_sha256':digest(off_path),
        'reference_sha256':on['reference_sha256'],'longitudinal_fit_sha256':on['longitudinal_fit_sha256'],
        'rms_speed_on_mps':on_rms,'rms_speed_off_mps':off_rms,
        'rms_speed_improvement_fraction':(off_rms-on_rms)/off_rms if off_rms>0 else None,
        'max_speed_error_on_mps':metrics_on['max_absolute_speed_error_mps'],
        'max_speed_error_off_mps':metrics_off['max_absolute_speed_error_mps'],
        'rms_position_on_m':pos_on['rms_position_error_m'],
        'rms_position_off_m':pos_off['rms_position_error_m'],
        'final_position_on_m':pos_on['final_position_error_m'],
        'final_position_off_m':pos_off['final_position_error_m'],
        'nonzero_throttle_corrections_on':metrics_on['nonzero_throttle_correction_samples'],
        'nonzero_brake_corrections_on':metrics_on['nonzero_brake_correction_samples'],
        'limitations':['Single matched pair is not statistical significance or drivetrain identification',
            'Unfitted reference speed profile includes discontinuous high braking demand',
            'Steering feedback is OFF in both arms; still subject to CARLA run-to-run effects']}
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
    p.add_argument('--longitudinal-fit',type=Path,default=DEFAULT_FIT)
    p.add_argument('--positive-summary',type=Path,default=fit.POS/'summary.json')
    p.add_argument('--positive-samples',type=Path,default=fit.POS/'calibration_samples.csv')
    p.add_argument('--negative-summary',type=Path,default=fit.NEG/'summary.json')
    p.add_argument('--negative-samples',type=Path,default=fit.NEG/'calibration_samples.csv')
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--inspect-server',action='store_true')
    mode.add_argument('--execute-track',action='store_true')
    mode.add_argument('--compare',action='store_true')
    p.add_argument('--confirm-drive',action='store_true')
    p.add_argument('--speed-feedback',choices=['on','off'])
    p.add_argument('--feedback-on-report',type=Path)
    p.add_argument('--feedback-off-report',type=Path)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=2000)
    p.add_argument('--timeout',type=float,default=10.)
    a=p.parse_args(argv)
    if a.confirm_drive != a.execute_track:p.error('CARLA movement needs --execute-track AND --confirm-drive')
    if a.execute_track and a.speed_feedback not in ('on','off'):p.error('Select --speed-feedback on or off')
    if not a.execute_track and a.speed_feedback is not None:p.error('--speed-feedback applies only to driving mode')
    if a.compare!=(a.feedback_on_report is not None and a.feedback_off_report is not None):
        p.error('--compare requires two --feedback-*-report files, and they only apply to --compare')
    if not math.isfinite(a.timeout) or a.timeout<=0:p.error('Timeout must be positive and finite')
    try:
        obs=b.read_stable_observation(a.observation)
        model=gen.validated_model(a.model,obs['sha256'])
        geometry=gen.geometry_from_project(a.route,a.map)
        reference=gen.load_reference(a.reference,obs,model,a.model,a.route,a.map,a.source_summary,geometry)
        reference['_loaded_from']=str(a.reference)
        fit_doc=fit.validate_fit(a.longitudinal_fit,a.positive_summary,a.positive_samples,a.negative_summary,a.negative_samples)
        if fit_doc['observation_sha256']!=obs['sha256']:raise ValueError('Longitudinal fit observation differs from CommonRoad reference observation')
        print('M6B-6: verified route/reference hashes, stable observation, and original 6B-3 longitudinal source CSVs')
        if a.compare:
            result=compare(a.feedback_on_report,a.feedback_off_report,a.out,reference)
            print('OFFLINE PAIRED COMPARISON:',result['conclusion'],'output:',a.out);return 0
        if a.execute_track and a.out.exists():raise FileExistsError('Refusing to overwrite previous CARLA run directory')
        if a.inspect_server:
            *_,pre=b.preflight(a,obs,drive=False)
            print('READ ONLY PREFLIGHT:',json.dumps(pre,indent=2));return 0
        if not a.execute_track:
            print('OFFLINE ONLY: No CARLA connection/tick/control. Route reference valid.');return 0
        return execute(a,obs,model,reference,geometry,fit_doc)
    except (OSError,ValueError,RuntimeError,TypeError,KeyError,ImportError) as exc:
        print('M6B-6 FAILED SAFE:',type(exc).__name__+': '+str(exc));return 2

if __name__=='__main__':raise SystemExit(main())
