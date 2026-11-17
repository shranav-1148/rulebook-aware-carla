"""Milestone 6B-1: CARLA observations -> CommonRoad-style planner state (NO driving).

Default: validate offline plan only. --inspect-server: read-only map/actor check.
--observe-existing: read-only snapshot of explicitly identified existing actors.
--capture-spawn --confirm-spawn: temporarily SPAWN ego+prop, passively observe one
CARLA frame, then destroy only those actors. Never world.tick(), apply_control(),
change world settings, seed ego velocity, or claim the snapshot is replanning-ready.

Original student-project integration code, NOT the upstream CommonRoad Rulebook.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

from carla_obstacle_m6a import build_plan, read_route, DEFAULT_EXPERIMENT, DEFAULT_ROUTE

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / 'output/carla_observation_6b1'
DEFAULT_BLUEPRINT_EGO = 'vehicle.tesla.model3'
DEFAULT_BLUEPRINT_PROP = 'static.prop.constructioncone'


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def finite(name, x):
    try:
        v = float(x)
    except (TypeError, ValueError) as e:
        raise ValueError(f'{name} is not numeric') from e
    if not math.isfinite(v):
        raise ValueError(f'{name} is not finite')
    return v


def cr_from_carla(x, y):
    return (finite('x',x), -finite('y',y))


def heading_cr_from_carla(yaw_deg):
    return wrap(-math.radians(finite('yaw_deg',yaw_deg)))


def body_center_carla(transform, local_bbox_center):
    yaw = math.radians(finite('yaw',transform.rotation.yaw))
    bx, by = finite('bbox_local_x',local_bbox_center.x), finite('bbox_local_y',local_bbox_center.y)
    return (float(transform.location.x) + bx*math.cos(yaw) - by*math.sin(yaw),
            float(transform.location.y) + bx*math.sin(yaw) + by*math.cos(yaw))


def actor_origin_for_body_center(cx, cy, yaw_deg, local_center):
    yaw=math.radians(yaw_deg)
    dx=local_center.x*math.cos(yaw)-local_center.y*math.sin(yaw)
    dy=local_center.x*math.sin(yaw)+local_center.y*math.cos(yaw)
    return cx-dx,cy-dy


def project_polyline(route, x, y):
    """Closest-segment station, signed offset (left-positive), tangent and distance."""
    best=None
    for a,b in zip(route,route[1:]):
        dx,dy=b[1]-a[1],b[2]-a[2]
        seg2=dx*dx+dy*dy
        if seg2<1e-14:
            continue
        t=max(0.,min(1.,((x-a[1])*dx+(y-a[2])*dy)/seg2))
        px,py=a[1]+t*dx,a[2]+t*dy
        dist2=(x-px)**2+(y-py)**2
        station=a[0]+t*(b[0]-a[0])
        signed_offset=(dx*(y-py)-dy*(x-px))/math.sqrt(seg2)
        entry=(dist2,station,signed_offset,math.atan2(dy,dx))
        if best is None or dist2<best[0]-1e-10 or (abs(dist2-best[0])<1e-10 and station>best[1]):
            best=entry
    if best is None:
        raise ValueError('Route has no nondegenerate segments')
    return {'station_m':best[1],'signed_lateral_offset_m':best[2],
            'tangent_yaw_cr_rad':best[3], 'distance_to_route_m':math.sqrt(best[0])}


def bbox_world(actor, pose):
    bb=actor.bounding_box
    verts=bb.get_world_vertices(pose)
    if len(verts)!=8:
        raise ValueError('Expected 8 CARLA bounding-box world vertices')
    vertices=[[finite('vx',v.x),finite('vy',v.y),finite('vz',v.z)] for v in verts]
    xmin,xmax=min(v[0] for v in vertices),max(v[0] for v in vertices)
    ymin,ymax=min(v[1] for v in vertices),max(v[1] for v in vertices)
    zmin,zmax=min(v[2] for v in vertices),max(v[2] for v in vertices)
    return {'vertices_carla_xyz_m':vertices,
            'vertices_commonroad_xy_m':[[v[0],-v[1]] for v in vertices],
            'bounds_carla_xyz_m':{'xmin':xmin,'xmax':xmax,'ymin':ymin,'ymax':ymax,'zmin':zmin,'zmax':zmax},
            'center_carla_xy_m':[(xmin+xmax)/2,(ymin+ymax)/2],
            'dimensions_world_aabb_m':[xmax-xmin,ymax-ymin,zmax-zmin],
            'local_center_xyz_m':[float(bb.location.x),float(bb.location.y),float(bb.location.z)],
            'local_half_extents_m':[float(bb.extent.x),float(bb.extent.y),float(bb.extent.z)]}


def first_reference_speed(csv_path):
    with Path(csv_path).open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames or not {'t_s','speed_mps'}.issubset(reader.fieldnames):
            raise ValueError('Missing expected bicycle trajectory columns')
        row=next(reader,None)
    if row is None or abs(finite('t_s',row['t_s']))>1e-8:
        raise ValueError('Missing t=0 reference')
    return finite('reference speed_mps',row['speed_mps'])


def offline_context(summary_path,route_path,safety_path,corridor_path):
    plan=build_plan(summary_path,route_path,safety_path,corridor_path)
    s=json.loads(Path(summary_path).read_text(encoding='utf-8'))['settings']
    required=['wheelbase_m','speed_min_mps','speed_max_mps','length_m','width_m',
              'max_abs_route_offset_m','max_abs_heading_error_deg','max_abs_steering_deg']
    settings={key:finite(key,s[key]) for key in required}
    if any(settings[k]<=0 for k in required):
        raise ValueError('Invalid planner model constants')
    target_speed=first_reference_speed(safety_path)
    other_speed=first_reference_speed(corridor_path)
    if abs(target_speed-other_speed)>1e-6:
        raise ValueError('Paired reference initial speeds are different')
    return {'plan':plan,'route':read_route(route_path),'settings':settings,
            'nominal_speed_mps':target_speed}


def snapshot_actor(snapshot, actor):
    snap=snapshot.find(actor.id)
    if snap is None:
        raise RuntimeError(f'Actor {actor.id} missing from world frame {snapshot.frame}; do not mix frames')
    return snap


def read_ego(snapshot,actor,carla):
    snap=snapshot_actor(snapshot,actor)
    pose=snap.get_transform()
    velocity=snap.get_velocity()
    accel=snap.get_acceleration()
    yaw_cr=heading_cr_from_carla(pose.rotation.yaw)
    cx,cy=body_center_carla(pose,actor.bounding_box.location)
    crx,cry=cr_from_carla(cx,cy)
    vx,vy,vz=finite('vx',velocity.x),finite('vy',velocity.y),finite('vz',velocity.z)
    vxcr,vycr=cr_from_carla(vx,vy)
    fwd=vxcr*math.cos(yaw_cr)+vycr*math.sin(yaw_cr)
    lat=-vxcr*math.sin(yaw_cr)+vycr*math.cos(yaw_cr)
    speed=math.sqrt(vx*vx+vy*vy+vz*vz)
    # CARLA's get_wheel_steer_angle() returns physics wheel angles (not control fraction).
    # Steering sign is NOT assumed to match the CommonRoad bicycle-model sign.
    wheel_angles=[finite('left_wheel_deg',actor.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)),
                  finite('right_wheel_deg',actor.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel))]
    return {'actor_id':int(actor.id),'type_id':str(actor.type_id),
            'actor_origin_carla_xyz_m':[float(pose.location.x),float(pose.location.y),float(pose.location.z)],
            'body_center_carla_xy_m':[cx,cy],'body_center_cr_xy_m':[crx,cry],
            'yaw_carla_deg':float(pose.rotation.yaw),'yaw_cr_rad':yaw_cr,
            'velocity_carla_xyz_mps':[vx,vy,vz],'velocity_cr_xy_mps':[vxcr,vycr],
            'acceleration_carla_xyz_mps2':[float(accel.x),float(accel.y),float(accel.z)],
            'total_speed_mps':speed,'longitudinal_speed_mps':fwd,'lateral_speed_mps':lat,
            'front_wheel_physics_angles_carla_deg':wheel_angles,
            'front_wheel_angle_mean_abs_deg':sum(abs(v) for v in wheel_angles)/2,
            'bbox':bbox_world(actor,pose)}


def read_obstacle(snapshot,actor):
    snap=snapshot_actor(snapshot,actor)
    pose=snap.get_transform()
    velocity=snap.get_velocity()
    bbox=bbox_world(actor,pose)
    cx,cy=bbox['center_carla_xy_m']
    return {'actor_id':int(actor.id),'type_id':str(actor.type_id),
            'actor_origin_carla_xyz_m':[float(pose.location.x),float(pose.location.y),float(pose.location.z)],
            'center_carla_xy_m':[cx,cy], 'center_cr_xy_m':[cx,-cy],
            'velocity_carla_xyz_mps':[float(velocity.x),float(velocity.y),float(velocity.z)],
            'bbox':bbox,'collision_mesh_verified':False}


def build_observation(snapshot,ego_actor,obs_actor,carla,context):
    """Snapshot-coherent observation; no runtime planner calls or control commands."""
    if ego_actor.id == obs_actor.id:
        raise ValueError('Ego and obstacle must be different actors')
    if not str(ego_actor.type_id).startswith('vehicle.') or not str(obs_actor.type_id).startswith('static.prop.'):
        raise ValueError('Expected ego vehicle and static.prop.* obstacle')
    ego=read_ego(snapshot,ego_actor,carla)
    obs=read_obstacle(snapshot,obs_actor)
    plan=context['plan']; model=context['settings']; route=context['route']
    ecx,ecy=ego['body_center_cr_xy_m']
    ocx,ocy=obs['center_cr_xy_m']
    proj=project_polyline(route,ecx,ecy)
    op=project_polyline(route,ocx,ocy)
    rear_x=ecx-0.5*model['wheelbase_m']*math.cos(ego['yaw_cr_rad'])
    rear_y=ecy-0.5*model['wheelbase_m']*math.sin(ego['yaw_cr_rad'])
    pos_err=math.hypot(ecx-plan['ego_initial_commonroad_body_center_xy_m'][0],
                       ecy-plan['ego_initial_commonroad_body_center_xy_m'][1])
    expected_yaw=-math.radians(plan['ego_initial_carla_yaw_deg'])
    heading_err_deg=abs(math.degrees(wrap(ego['yaw_cr_rad']-expected_yaw)))
    obs_err=math.hypot(ocx-plan['obstacle_model']['commonroad_xy_m'][0],
                       ocy-plan['obstacle_model']['commonroad_xy_m'][1])
    route_heading_error=abs(math.degrees(wrap(ego['yaw_cr_rad']-proj['tangent_yaw_cr_rad'])))
    # Zero is the only unambiguous initial wheel-steer condition without a
    # calibrated CARLA-wheel-steering-sign convention and axle model.
    near_zero_steering=ego['front_wheel_angle_mean_abs_deg'] < 0.5
    flags={
        'ego_pose_matches_nominal_start':pos_err <= 0.25 and heading_err_deg<=3.,
        'obstacle_center_matches_offline_location':obs_err<=0.10,
        'ego_within_route_offset_bound':abs(proj['signed_lateral_offset_m'])<=model['max_abs_route_offset_m'],
        'ego_within_route_heading_bound':route_heading_error<=model['max_abs_heading_error_deg'],
        'forward_speed_within_planner_bounds':model['speed_min_mps']<=ego['longitudinal_speed_mps']<=model['speed_max_mps'],
        'small_lateral_velocity':abs(ego['lateral_speed_mps'])<=0.5,
        'initial_wheel_steering_near_zero':near_zero_steering,
        'physical_obstacle_matches_offline_circle':False,
        'live_bicycle_reinitialization_implemented':False}
    candidate={'rear_axle_x_cr_m':rear_x,'rear_axle_y_cr_m':rear_y,
               'yaw_cr_rad':ego['yaw_cr_rad'],
               'forward_speed_mps':ego['longitudinal_speed_mps'],
               'steer_rad':0.0 if near_zero_steering else None,
               'steer_source':'near-zero wheel physics only' if near_zero_steering else 'unverified wheel steering sign',
               'body_center_projection':proj,
               'geometry_assumption':'body-center is 0.5 * assumed wheelbase ahead of rear axle; rear axle not observed directly',
               'vehicle_4eb_field_mapping':{'x':'rear_axle_x_cr_m','y':'rear_axle_y_cr_m','yaw':'yaw_cr_rad','v':'forward_speed_mps','steer':'steer_rad'},
               'ready_for_original_4eb_build_motion_tree':False,
               'ready_for_live_replanning':False,
               'not_ready_reasons':([k for k,v in flags.items() if not v] +
                                    ['4E-B build_motion_tree still constructs a fixed t=0 initial_vehicle, not an observed initial Vehicle'])}
    t=snapshot.timestamp
    initial_ref_speed=context.get('nominal_speed_mps')
    return {'milestone':'6B-1','snapshot':{'frame':int(snapshot.frame),
                        'elapsed_seconds':float(t.elapsed_seconds),'delta_seconds':float(t.delta_seconds)},
            'ego':ego,'obstacle':obs,'ego_route_projection':proj,'obstacle_route_projection':op,
            'errors_vs_offline':{'ego_body_center_position_m':pos_err,'ego_heading_deg':heading_err_deg,
                                 'nominal_reference_speed_mps':initial_ref_speed,
                                 'observed_minus_reference_forward_speed_mps':(ego['longitudinal_speed_mps']-initial_ref_speed if initial_ref_speed is not None else None),
                                 'obstacle_center_position_m':obs_err,
                                 'observed_obstacle_center_to_ego_center_m':math.hypot(ocx-ecx,ocy-ecy)},
            'offline_obstacle':plan['obstacle_model'],
            'planner_initial_state_candidate':candidate,'checks':flags,
            'model_note':'CARLA prop bounding box and circle are not equivalent; soft clearance margin is NOT physical obstacle size.',
            'behavior_note':'No steering/throttle/brake, no replanning. A parked ego is not a valid dynamic start at 5-8m/s.'}


def save_json(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return path


def connect(args, context, need_blueprints=False):
    try:
        import carla
    except ImportError as e:
        raise RuntimeError('Use CARLA 0.9.16 Windows Python .venv') from e
    client=carla.Client(args.host,args.port)
    client.set_timeout(args.timeout)
    world=client.get_world()
    cv,sv=str(client.get_client_version()),str(client.get_server_version())
    if cv!='0.9.16' or sv!='0.9.16':
        raise RuntimeError('CARLA version mismatch '+cv+' / '+sv)
    wm=world.get_map()
    if wm.name.split('/')[-1]!='Town01':
        raise RuntimeError('CARLA map is '+wm.name+', expected Town01')
    cx,cy=context['plan']['ego_initial_carla_body_center_xy_m']
    wp=wm.get_waypoint(carla.Location(cx,cy,0), project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        raise RuntimeError('No CARLA driving waypoint near initial ego')
    dx,dy=cx-wp.transform.location.x,cy-wp.transform.location.y
    yaw_deg=context['plan']['ego_initial_carla_yaw_deg']
    yaw_diff=abs(math.degrees(wrap(math.radians(yaw_deg-wp.transform.rotation.yaw))))
    if math.hypot(dx,dy)>1.75 or yaw_diff>30:
        raise RuntimeError('Start pose does not align with Town01 road')
    actors=world.get_actors()
    existing_vehicles=list(actors.filter('vehicle.*'))
    info={'carla_client_version':cv,'carla_server_version':sv,'map':wm.name,
          'world_synchronous':bool(world.get_settings().synchronous_mode),
          'existing_vehicle_count':len(existing_vehicles),
          'start_waypoint_xy_distance_m':math.hypot(dx,dy),
          'start_waypoint_heading_error_deg':yaw_diff,
          'actions_performed':'read only'}
    blueprints=None
    if need_blueprints:
        bps=world.get_blueprint_library()
        blueprints=(bps.find(args.ego_blueprint),bps.find(args.obstacle_blueprint))
    return carla,world,wp,info,blueprints


def capture_existing(args,context,carla,world,info):
    if args.ego_id == args.obstacle_id:
        raise ValueError('Ego ID equals obstacle ID')
    ego=world.get_actor(args.ego_id)
    obs=world.get_actor(args.obstacle_id)
    if ego is None or obs is None:
        raise RuntimeError('Specified actor ID does not exist')
    snap=world.get_snapshot()  # read-only, do not tick an externally-controlled world
    report=build_observation(snap,ego,obs,carla,context)
    report['server']=info
    report['observation_mode']='read_only_existing_actors'
    report['owned_actor_ids']=[]
    report['client_world_ticks']=0
    return report


def spawn_capture(args,context,carla,world,wp,info,blueprints):
    if not args.confirm_spawn:
        raise RuntimeError('Explicit --confirm-spawn required')
    if info['world_synchronous'] or info['existing_vehicle_count']:
        raise RuntimeError('Refuse actor creation in synchronous or vehicle-occupied world')
    ego_bp,obstacle_bp=blueprints
    if not ego_bp.id.startswith('vehicle.') or not obstacle_bp.id.startswith('static.prop.'):
        raise RuntimeError('Unexpected ego/obstacle blueprint type')
    cx,cy=context['plan']['ego_initial_carla_body_center_xy_m']
    yaw=context['plan']['ego_initial_carla_yaw_deg']
    ox,oy=context['plan']['obstacle_model']['carla_xy_m']
    road_z=wp.transform.location.z
    obs_wp=world.get_map().get_waypoint(carla.Location(ox,oy,0),project_to_road=True,lane_type=carla.LaneType.Driving)
    if obs_wp is None:
        raise RuntimeError('Obstacle projected waypoint unavailable')
    ego=None; prop=None
    result=None
    cleanup={}
    try:
        ego=world.try_spawn_actor(ego_bp,carla.Transform(carla.Location(cx,cy,road_z+args.ego_z_offset),carla.Rotation(yaw=yaw)))
        if ego is None:
            raise RuntimeError('Could not spawn temporary ego vehicle')
        # Exactly one pose correction; uses CARLA box center as body center, same as 5A.
        center_local=ego.bounding_box.location
        actor_x,actor_y=actor_origin_for_body_center(cx,cy,yaw,center_local)
        ego.set_transform(carla.Transform(carla.Location(actor_x,actor_y,road_z+args.ego_z_offset),carla.Rotation(yaw=yaw)))
        # No set_autopilot, no speed seeding, and no apply_control().
        # Offset from road waypoint Z derived as in 6A; NOT verified terrain height.
        prop=world.try_spawn_actor(obstacle_bp,carla.Transform(carla.Location(ox,oy,obs_wp.transform.location.z+3.0),carla.Rotation(yaw=0)))
        if prop is None:
            raise RuntimeError('Could not spawn temporary obstacle prop')
        before=bbox_world(prop,prop.get_transform())
        bx,by=before['center_carla_xy_m']
        zbottom=before['bounds_carla_xyz_m']['zmin']
        pose=prop.get_transform()
        prop.set_transform(carla.Transform(
            carla.Location(pose.location.x+ox-bx,pose.location.y+oy-by,
                           pose.location.z+obs_wp.transform.location.z+0.02-zbottom),pose.rotation))
        # Passive wait for a published frame with BOTH actors. Never world.tick().
        snap=None
        for _ in range(4):
            snap=world.wait_for_tick(args.timeout)
            if snap.find(ego.id) is not None and snap.find(prop.id) is not None:
                break
        if snap is None or snap.find(ego.id) is None or snap.find(prop.id) is None:
            raise RuntimeError('No coherent CARLA frame containing both newly spawned actors')
        result=build_observation(snap,ego,prop,carla,context)
        result['server']=info
        result['observation_mode']='temporary_spawn_passive_snapshot'
        result['owned_actor_ids']=[int(ego.id),int(prop.id)]
        result['client_world_ticks']=0
        result['ego_initial_velocity_seeded']=False
        result['obstacle_waypoint_z_source']='nearest driving lane, terrain height not independently verified'
        return result
    finally:
        for name,actor in [('obstacle',prop),('ego',ego)]:
            if actor is not None:
                try:
                    cleanup[name]={'actor_id':int(actor.id),'destroy_returned_success':bool(actor.destroy())}
                except Exception as e:
                    cleanup[name]={'actor_id':int(actor.id),'destroy_returned_success':False,'error':str(e)}
        if cleanup:
            print('TEMPORARY ACTOR CLEANUP:',cleanup)
        if result is not None:
            result['cleanup']=cleanup
            result['cleanup_ok']=all(v['destroy_returned_success'] for v in cleanup.values()) and len(cleanup)==2


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,default=DEFAULT_EXPERIMENT/'summary.json')
    p.add_argument('--route',type=Path,default=DEFAULT_ROUTE)
    p.add_argument('--safety-csv',type=Path,default=DEFAULT_EXPERIMENT/'safety_first_bicycle_samples.csv')
    p.add_argument('--corridor-csv',type=Path,default=DEFAULT_EXPERIMENT/'corridor_first_bicycle_samples.csv')
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--inspect-server',action='store_true')
    modes.add_argument('--observe-existing',action='store_true',help='Read only; requires explicit actor IDs')
    modes.add_argument('--capture-spawn',action='store_true',help='Spawn two temporary actors; NO ego driving')
    p.add_argument('--confirm-spawn',action='store_true',help='Required for --capture-spawn')
    p.add_argument('--ego-id',type=int)
    p.add_argument('--obstacle-id',type=int)
    p.add_argument('--ego-blueprint',default=DEFAULT_BLUEPRINT_EGO)
    p.add_argument('--obstacle-blueprint',default=DEFAULT_BLUEPRINT_PROP)
    p.add_argument('--ego-z-offset',type=float,default=0.5)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=2000)
    p.add_argument('--timeout',type=float,default=8.)
    args=p.parse_args(argv)
    if args.confirm_spawn and not args.capture_spawn:
        p.error('--confirm-spawn requires --capture-spawn')
    if args.observe_existing:
        if args.ego_id is None or args.obstacle_id is None:
            p.error('--observe-existing requires both --ego-id and --obstacle-id')
    elif args.ego_id is not None or args.obstacle_id is not None:
        p.error('Actor IDs are only allowed with --observe-existing')
    if not all(math.isfinite(v) for v in (args.timeout,args.ego_z_offset)) or args.timeout<=0 or not 0<=args.ego_z_offset<=1.5:
        p.error('Invalid timeout or ego spawn height offset')
    if args.port<1 or args.port>65535:
        p.error('Invalid server port')
    try:
        ctx=offline_context(args.summary,args.route,args.safety_csv,args.corridor_csv)
        plan=ctx['plan']
        print('===== MILESTONE 6B-1: OFFLINE PREFLIGHT =====')
        print('Nominal ego center CommonRoad:',*[round(v,3) for v in plan['ego_initial_commonroad_body_center_xy_m']])
        print('Nominal obstacle CommonRoad:',*[round(v,3) for v in plan['obstacle_model']['commonroad_xy_m']])
        print('Offline model obstacle radius:',plan['obstacle_model']['radius_m'],'m; CARLA cone is NOT geometrically equivalent')
        if not (args.inspect_server or args.observe_existing or args.capture_spawn):
            print('OFFLINE PASS; no CARLA connection, actor modification or ticks.')
            return 0
        carla,world,wp,info,bps=connect(args,ctx,need_blueprints=args.capture_spawn)
        print('CARLA:',info['carla_client_version'],'/',info['carla_server_version'],'map:',info['map'])
        print('Initial lane distance: %.4f m; heading difference: %.3f deg' %
              (info['start_waypoint_xy_distance_m'],info['start_waypoint_heading_error_deg']))
        if args.inspect_server:
            info['action']='read_only_preflight'
            path=save_json(args.out/'server_preflight.json',info)
            print('READ-ONLY SERVER PREFLIGHT:',path)
            return 0
        target=args.out/'live_observation.json'
        if target.exists():
            raise RuntimeError('Output already exists; choose a fresh --out directory to preserve earlier snapshots')
        if args.capture_spawn:
            report=spawn_capture(args,ctx,carla,world,wp,info,bps)
        else:
            report=capture_existing(args,ctx,carla,world,info)
        path=save_json(target,report)
        print('Snapshot frame:',report['snapshot']['frame'],'ego id:',report['ego']['actor_id'],
              'obstacle id:',report['obstacle']['actor_id'])
        print('Ego speed measured:',round(report['ego']['longitudinal_speed_mps'],3),'m/s; speed admissible:',
              report['checks']['forward_speed_within_planner_bounds'])
        print('Ego pose error: %.4f m; obstacle placement error: %.4f m' %
              (report['errors_vs_offline']['ego_body_center_position_m'],
               report['errors_vs_offline']['obstacle_center_position_m']))
        print('Planner-ready:',report['planner_initial_state_candidate']['ready_for_live_replanning'],
              '(6B-2 must implement observed-state initialization and measured obstacle geometry)')
        if 'cleanup_ok' in report:
            print('Created actors removed:',report['cleanup_ok'])
        print('Observation:',path)
        return 0 if report.get('cleanup_ok',True) else 3
    except (OSError,ValueError,RuntimeError,KeyError,AttributeError) as e:
        print('ERROR:',e)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
