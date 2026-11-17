"""Milestone 6A: reproduce and inspect a synthetic stationary obstacle in CARLA Town01.

Default: OFFLINE preflight (no CARLA import). --inspect-server: read-only CARLA.
--spawn-inspect --confirm-spawn: temporary STATIC PROP; no ego, no world ticks,
no replanning, no collision claims, actor is destroyed in finally.
This is a geometry/scene-registry experiment, not obstacle-avoidance validation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from bisect import bisect_right
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT = HERE / 'output/town01_bicycle_graph_beam800_min40'
DEFAULT_ROUTE = HERE / 'output/town01_graph/route_waypoints.csv'
DEFAULT_OUT = HERE / 'output/carla_obstacle_6a'
DEFAULT_BLUEPRINT = 'static.prop.constructioncone'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(name, value):
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} is not numeric') from exc
    if not math.isfinite(x):
        raise ValueError(f'{name} is not finite')
    return x


def read_route(path):
    """Match the M4B route deduplication and cumulative arc-length algorithm."""
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2:
        raise ValueError('Route requires at least two waypoint rows')
    pts = []
    for row in rows:
        x, y = finite('route.x_m', row.get('x_m')), finite('route.y_m', row.get('y_m'))
        if not pts or math.hypot(x-pts[-1][1], y-pts[-1][2]) > 1e-8:
            s = 0.0 if not pts else pts[-1][0] + math.hypot(x-pts[-1][1],y-pts[-1][2])
            pts.append((s,x,y))
    if len(pts) < 2 or pts[-1][0] <= 0:
        raise ValueError('Degenerate reference route')
    return pts


def route_pose(pts, station):
    station = finite('obstacle_station_m', station)
    if not 0 <= station <= pts[-1][0]:
        raise ValueError('Obstacle station outside reference route')
    ss = [p[0] for p in pts]
    i = max(0,min(bisect_right(ss,station)-1,len(pts)-2))
    a,b=pts[i],pts[i+1]
    f=(station-a[0])/(b[0]-a[0])
    x=a[1]+f*(b[1]-a[1])
    y=a[2]+f*(b[2]-a[2])
    yaw=math.atan2(b[2]-a[2],b[1]-a[1])
    return x,y,yaw


def route_offset_xy(pts, station, d):
    x,y,yaw=route_pose(pts,station)
    d=finite('obstacle_offset_m',d)
    return x-math.sin(yaw)*d,y+math.cos(yaw)*d


def read_start(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames or not all(k in reader.fieldnames for k in ('t_s','x_m','y_m','yaw_rad')):
            raise ValueError('Not a Milestone 4E-B bicycle-samples CSV')
        row=next(reader,None)
    if row is None:
        raise ValueError('Empty trajectory CSV')
    t=finite('t_s',row.get('t_s'))
    if abs(t)>1e-8:
        raise ValueError('Trajectory does not begin at t=0')
    return tuple(finite(k,row.get(k)) for k in ('x_m','y_m','yaw_rad'))


def build_plan(summary_path, route_path, safety_path, corridor_path):
    summary_path,route_path=Path(summary_path),Path(route_path)
    doc=json.loads(summary_path.read_text(encoding='utf-8'))
    if doc.get('route_source')=='synthetic-demo':
        raise ValueError('Refusing synthetic-demo data; requires the Town01 experiment')
    settings=doc.get('settings') or {}
    required=('obstacle_station_m','obstacle_offset_m','obstacle_radius_m','clearance_margin_m')
    if not all(k in settings for k in required):
        raise ValueError('M4E-B summary missing obstacle settings')
    s,d,r,soft=(finite(k,settings[k]) for k in required)
    if r<=0 or soft<0:
        raise ValueError('Invalid obstacle radius/clearance margin')
    route_hash=sha256(route_path)
    if doc.get('route_sha256')!=route_hash:
        raise ValueError('Route SHA256 differs from M4E-B experiment; do not silently substitute route')
    pts=read_route(route_path)
    cx,cy=route_offset_xy(pts,s,d)
    stored=doc.get('obstacle_xy_m')
    if not isinstance(stored,list) or len(stored)!=2:
        raise ValueError('M4E-B summary missing obstacle_xy_m')
    sx,sy=map(float,stored)
    if math.hypot(cx-sx,cy-sy)>0.02:
        raise ValueError('Stored obstacle position does not match route and settings')
    safety=read_start(safety_path)
    corridor=read_start(corridor_path)
    delta=math.hypot(safety[0]-corridor[0],safety[1]-corridor[1])
    yaw_delta=abs(math.atan2(math.sin(safety[2]-corridor[2]),math.cos(safety[2]-corridor[2])))
    if delta>0.05 or yaw_delta>math.radians(0.5):
        raise ValueError('R9/R5 CSVs do not have the same nominal initial pose')
    center_x,center_y,route_yaw=route_pose(pts,s)
    return {
        'experiment':'Milestone 6A stationary obstacle scene preflight',
        'status':'offline_geometry_verified',
        'source_summary':str(summary_path), 'source_summary_sha256':sha256(summary_path),
        'source_route':str(route_path),'source_route_sha256':route_hash,
        'source_safety':str(safety_path),'source_safety_sha256':sha256(safety_path),
        'source_corridor':str(corridor_path),'source_corridor_sha256':sha256(corridor_path),
        'planning_priority_type':'AI-authored Rulebook-inspired proxies; not original Rulebook',
        'obstacle_model':{'type':'stationary_circle_2d','station_m':s,'offset_m':d,
                          'radius_m':r,'soft_clearance_margin_m':soft,
                          'commonroad_xy_m':[cx,cy], 'carla_xy_m':[cx,-cy],
                          'reference_centerline_cr_xy_m':[center_x,center_y],
                          'reference_heading_cr_deg':math.degrees(route_yaw)},
        'ego_initial_commonroad_body_center_xy_m':[safety[0],safety[1]],
        'ego_initial_carla_body_center_xy_m':[safety[0],-safety[1]],
        'ego_initial_carla_yaw_deg':-math.degrees(safety[2]),
        'initial_body_center_to_obstacle_center_distance_m':math.hypot(safety[0]-cx,safety[1]-cy),
        'warning':'The physical prop is NOT guaranteed to match a circle or possess a collision mesh; no vehicle execution or live planning.'}


def projected_bbox_geometry(actor):
    """Measure actual WORLD bounding box, not merely blueprint attributes."""
    bb=actor.bounding_box
    vertices=bb.get_world_vertices(actor.get_transform())
    if len(vertices)!=8:
        raise ValueError('Actor does not expose an eight-vertex bounding box')
    xs=[finite('box vertex x',v.x) for v in vertices]
    ys=[finite('box vertex y',v.y) for v in vertices]
    zs=[finite('box vertex z',v.z) for v in vertices]
    return {'bounds_world_xyz_m':{'xmin':min(xs),'xmax':max(xs),
                                  'ymin':min(ys),'ymax':max(ys),
                                  'zmin':min(zs),'zmax':max(zs)},
            'world_aabb_center_xy_m':[(min(xs)+max(xs))/2,(min(ys)+max(ys))/2],
            'world_aabb_extent_xy_m':[(max(xs)-min(xs))/2,(max(ys)-min(ys))/2],
            'bbox_local_center_m':[float(bb.location.x),float(bb.location.y),float(bb.location.z)],
            'bbox_local_extents_m':[float(bb.extent.x),float(bb.extent.y),float(bb.extent.z)]}


def shape_report(actual, plan):
    """An axis-aligned bbox is NOT the underlying circular model or collision mesh."""
    ex,ey,ez=actual['bbox_local_extents_m']
    r=plan['obstacle_model']['radius_m']
    if min(ex,ey,ez)<=1e-6:
        return {'valid_bbox':False,'geometry_match':False,
                'reason':'Zero/degenerate measured prop bbox; cannot compare physical dimensions'}
    return {'valid_bbox':True, 'geometry_match':False,
            'reason':'CARLA prop bbox is a 3D box; planner obstacle is a 2D circle. Exact occupancy is not equivalent.',
            'model_circle_radius_m':r,
            'prop_bbox_xy_full_widths_m':[2*ex,2*ey],
            'prop_bbox_xy_inscribed_radius_proxy_m':min(ex,ey),
            'prop_bbox_xy_corner_radius_proxy_m':math.hypot(ex,ey),
            'box_does_not_verify_collision_mesh':True}


def save_json(out, name, data):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    path=out/name
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return path


def carla_preflight(args,plan):
    try:
        import carla
    except ImportError as exc:
        raise RuntimeError('Install/use matching CARLA 0.9.16 Python API in Windows .venv') from exc
    client=carla.Client(args.host,args.port)
    client.set_timeout(args.timeout)
    world=client.get_world()
    cv,sv=str(client.get_client_version()),str(client.get_server_version())
    if '0.9.16' not in cv or '0.9.16' not in sv:
        raise RuntimeError('Expected CARLA client and server 0.9.16; got '+cv+' / '+sv)
    m=world.get_map()
    if m.name.split('/')[-1]!='Town01':
        raise RuntimeError('Wrong CARLA map '+m.name+' (expected Town01)')
    bps=world.get_blueprint_library()
    if args.list_blueprints:
        choices=sorted(b.id for b in bps.filter('static.prop.*'))
        print('Available static.prop.* blueprints:',len(choices))
        for bp_id in choices:
            if any(x in bp_id for x in ('cone','barrier','warning','box')):
                print('  ',bp_id)
    try:
        bp=bps.find(args.blueprint)
    except Exception as exc:
        raise RuntimeError('Prop blueprint unavailable: '+args.blueprint) from exc
    if not bp.id.startswith('static.prop.'):
        raise RuntimeError('Refusing non-static-prop blueprint for Milestone 6A')
    cx,cy=plan['obstacle_model']['carla_xy_m']
    wp=m.get_waypoint(carla.Location(cx,cy,0),project_to_road=True,lane_type=carla.LaneType.Driving)
    if wp is None:
        raise RuntimeError('No nearby projected Town01 driving waypoint')
    dist=math.hypot(cx-wp.transform.location.x,cy-wp.transform.location.y)
    existing_vehicles=list(world.get_actors().filter('vehicle.*'))
    existing_props=list(world.get_actors().filter('static.prop.*'))
    sync=bool(world.get_settings().synchronous_mode)
    report={'status':'server_inspected','client_version':cv,'server_version':sv,
            'map':m.name,'blueprint_id':bp.id,
            'waypoint_xy_m':[wp.transform.location.x,wp.transform.location.y],
            'waypoint_z_m':wp.transform.location.z,
            'obstacle_offset_from_projected_lane_center_m':dist,
            'existing_vehicle_count':len(existing_vehicles),
            'existing_spawned_prop_count':len(existing_props),
            'world_already_synchronous':sync,
            'no_world_ticks_issued':True,
            'warning':'Waypoint Z at projected road lane can differ from actual terrain height at off-lane obstacle.'}
    print('CARLA version:',cv,'/',sv,'map:',m.name)
    print('Offline obstacle CARLA XY: ({:.3f}, {:.3f}); projected road waypoint lateral distance: {:.3f} m'.format(cx,cy,dist))
    print('Blueprint:',bp.id,'existing vehicles:',len(existing_vehicles),'world synchronous:',sync)
    return carla,world,bp,wp,report


def spawn_and_inspect(args,plan,carla,world,bp,wp,report):
    if not args.confirm_spawn:
        raise RuntimeError('--spawn-inspect requires --confirm-spawn (actor creation is not read-only)')
    if report['existing_vehicle_count'] or report['world_already_synchronous']:
        raise RuntimeError('Refusing temporary spawn in an occupied/synchronous world. Use an isolated asynchronous Town01 instance.')
    cx,cy=plan['obstacle_model']['carla_xy_m']
    road_z=float(wp.transform.location.z)
    actor=None
    spawned=False
    result=dict(report)
    try:
        if bp.has_attribute('role_name'):
            bp.set_attribute('role_name','milestone6a_inspection')
        provisional=carla.Transform(carla.Location(cx,cy,road_z+3.0),carla.Rotation(yaw=args.prop_yaw))
        actor=world.try_spawn_actor(bp,provisional)
        if actor is None:
            raise RuntimeError('Prop could not spawn at provisional position')
        spawned=True
        # Move the real measured box AABB center to the target XY and bottom to road Z.
        before=projected_bbox_geometry(actor)
        bx,by=before['world_aabb_center_xy_m']
        bottom=before['bounds_world_xyz_m']['zmin']
        pose=actor.get_transform()
        corrected=carla.Transform(
            carla.Location(pose.location.x+(cx-bx),pose.location.y+(cy-by),
                           pose.location.z+(road_z+args.ground_clearance-bottom)),
            pose.rotation)
        actor.set_transform(corrected)   # ONE placement correction; no simulated vehicle movement
        # In async mode, give server one moment to process the transform; NO client world.tick().
        time.sleep(0.3)
        measured=projected_bbox_geometry(actor)
        mx,my=measured['world_aabb_center_xy_m']
        xy_err=math.hypot(cx-mx,cy-my)
        base_err=measured['bounds_world_xyz_m']['zmin']-(road_z+args.ground_clearance)
        result.update({'status':'spawned_measured_cleanup_pending',
                       'temporary_actor_id':int(actor.id),'temporary_actor_type':str(actor.type_id),
                       'measured_bbox':measured,'proxy_shape_comparison':shape_report(measured,plan),
                       'actual_obstacle_bbox_center_cr_xy_m':[mx,-my],
                       'placement_xy_error_m':xy_err,
                       'placement_bottom_z_error_m':base_err,
                       'road_height_source':'projected-nearest-driving-waypoint',
                       'collision_mesh_verified':False,
                       'simulation_tick_called_by_script':False})
        print('TEMPORARY PROP ID:',actor.id,'type:',actor.type_id)
        print('World bbox full dimensions (X,Y,Z): {:.3f}, {:.3f}, {:.3f} m'.format(
            2*measured['world_aabb_extent_xy_m'][0],2*measured['world_aabb_extent_xy_m'][1],
            measured['bounds_world_xyz_m']['zmax']-measured['bounds_world_xyz_m']['zmin']))
        print('Position offset vs offline obstacle: {:.4f}m; box bottom relative to waypoint Z: {:.4f}m'.format(xy_err,base_err))
        print('GEOMETRY EQUIVALENCE: NOT ESTABLISHED (box versus circle; unknown collision mesh)')
        if xy_err>0.10 or abs(base_err)>0.10:
            print('WARNING: measured placement tolerance >0.10m; review before next milestone')
        if args.hold_seconds>0:
            print('Temporary prop visible for {:.1f} s; cleanup follows automatically.'.format(args.hold_seconds))
            time.sleep(args.hold_seconds)
        return result
    finally:
        if actor is not None:
            try:
                success=bool(actor.destroy())
                print('Temporary prop destroyed:',success)
                result['temporary_actor_destroy_returned_success']=success
                result['status']='spawned_measured_then_destroyed' if success else 'CLEANUP_UNCONFIRMED'
            except Exception as exc:
                result['temporary_actor_destroy_returned_success']=False
                result['status']='CLEANUP_FAILED'
                result['cleanup_error']=str(exc)
                print('WARNING: failed to destroy temporary actor',exc)
            if 'measured_bbox' in result:
                save_json(args.out,'actor_inspection.json',result)
        if not spawned:
            print('No actor created; nothing to clean up.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,default=DEFAULT_EXPERIMENT/'summary.json')
    parser.add_argument('--route',type=Path,default=DEFAULT_ROUTE)
    parser.add_argument('--safety-csv',type=Path,default=DEFAULT_EXPERIMENT/'safety_first_bicycle_samples.csv')
    parser.add_argument('--corridor-csv',type=Path,default=DEFAULT_EXPERIMENT/'corridor_first_bicycle_samples.csv')
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    actions=parser.add_mutually_exclusive_group()
    actions.add_argument('--inspect-server',action='store_true',help='Connect to CARLA; read-only; no actors or ticks')
    actions.add_argument('--spawn-inspect',action='store_true',help='Temporarily create a static prop, measure it, destroy it')
    parser.add_argument('--confirm-spawn',action='store_true',help='Required acknowledgement for --spawn-inspect')
    parser.add_argument('--list-blueprints',action='store_true',help='List candidate static.prop.* IDs during server inspection')
    parser.add_argument('--blueprint',default=DEFAULT_BLUEPRINT)
    parser.add_argument('--prop-yaw',type=float,default=0.,help='CARLA yaw in degrees, purely for prop orientation')
    parser.add_argument('--ground-clearance',type=float,default=0.02,help='Box bottom height above projected road Z (m)')
    parser.add_argument('--hold-seconds',type=float,default=5.,help='How long to display the prop before automatic removal')
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=2000)
    parser.add_argument('--timeout',type=float,default=8.)
    args=parser.parse_args(argv)
    if not all(math.isfinite(x) for x in (args.prop_yaw,args.ground_clearance,args.hold_seconds,args.timeout)) or args.hold_seconds<0 or abs(args.ground_clearance)>1 or args.timeout<=0:
        parser.error('Invalid finite yaw/ground clearance/hold/timeout settings')
    if args.confirm_spawn and not args.spawn_inspect:
        parser.error('--confirm-spawn is only used with --spawn-inspect')
    if args.list_blueprints and not (args.inspect_server or args.spawn_inspect):
        parser.error('--list-blueprints needs a CARLA server inspection mode')
    try:
        plan=build_plan(args.summary,args.route,args.safety_csv,args.corridor_csv)
        print('===== MILESTONE 6A: OFFLINE OBSTACLE PREFLIGHT =====')
        m=plan['obstacle_model']
        print('Offline model: stationary circle radius {:.2f}m; station {:.1f}m, offset {:.1f}m'.format(m['radius_m'],m['station_m'],m['offset_m']))
        print('CommonRoad XY: ({:.3f}, {:.3f}); CARLA XY: ({:.3f}, {:.3f})'.format(*m['commonroad_xy_m'],*m['carla_xy_m']))
        print('Route hash and both nominal starting poses verified.')
        offline_path=save_json(args.out,'obstacle_plan.json',plan)
        print('Plan:',offline_path)
        if not args.inspect_server and not args.spawn_inspect:
            print('OFFLINE ONLY: no CARLA connection, world ticks or actors.')
            return 0
        carla,world,bp,wp,report=carla_preflight(args,plan)
        preflight_path=save_json(args.out,'server_preflight.json',report)
        print('Server preflight:',preflight_path)
        if args.inspect_server:
            print('READ-ONLY SERVER INSPECTION COMPLETE: no actor spawned, no world ticks.')
            return 0
        spawn_and_inspect(args,plan,carla,world,bp,wp,report)
        print('SPAWN INSPECTION COMPLETE: temporary prop removed; no ego or planner execution.')
        return 0
    except (OSError,ValueError,RuntimeError,KeyError) as exc:
        print('ERROR:',exc)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
