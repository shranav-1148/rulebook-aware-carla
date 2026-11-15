"""Milestone 4E-B: steering-rate-controlled kinematic-bicycle motion-tree prototype.

Project-authored *offline* motion planner. One shared time-layered, beam-pruned
motion tree for two lexicographic Rulebook-inspired PROXY priority profiles.
NOT original CommonRoad Rulebook, CARLA execution, or certified safe planning.

New physics state is (x, y, yaw, forward_speed, steering_angle), with explicit
controls (longitudinal acceleration, steering RATE). Each primitive is integrated
with RK4, with no position or heading snapping between consecutive edges.
Grid bins are used ONLY to pick representative candidates in a bounded search.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Optional

from shapely.geometry import Point, Polygon, box
from shapely.prepared import prep

from town01_motion_graph import read_route_csv, pose_at, synthetic_route, world_xy
from town01_feasibility_check import load_town01_road, vehicle_rectangle

HERE = Path(__file__).resolve().parent
DEFAULT_ROUTE = HERE / 'output/town01_graph/route_waypoints.csv'
DEFAULT_MAP = HERE / 'maps/Town01.rpp_compat.xml'
DEFAULT_OUT = HERE / 'output/town01_bicycle_graph'
PLANNED_LANELETS = (38, 40, 58, 60)
VALIDATION_LANELETS = (39, 38, 40, 58, 60)


def rad(degrees):
    return math.radians(degrees)


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class Config:
    dt_s: float = 1.0
    horizon_steps: int = 6
    integration_dt_s: float = 0.1
    speed_initial_mps: float = 6.0
    speed_min_mps: float = 5.0
    speed_max_mps: float = 8.0
    accelerations_mps2: tuple = (-1.0, 0.0, 1.0)
    steering_rates_deg_s: tuple = (-8.0, -4.0, 0.0, 4.0, 8.0)
    max_abs_steering_deg: float = 25.0
    wheelbase_m: float = 2.7
    max_lateral_accel_mps2: float = 3.0
    length_m: float = 4.5
    width_m: float = 1.9
    road_area_tolerance_m2: float = 1e-5
    max_abs_route_offset_m: float = 2.5
    max_abs_heading_error_deg: float = 35.0
    obstacle_station_m: float = 21.0
    obstacle_offset_m: float = -2.4
    obstacle_radius_m: float = 0.65
    clearance_margin_m: float = 1.6
    # Resource-bounded search. Bin values do not SNAP simulated states.
    station_bin_m: float = 2.0
    lateral_bin_m: float = 0.25
    heading_bin_deg: float = 5.0
    speed_bin_mps: float = 1.0
    steering_bin_deg: float = 4.0
    beam_width: int = 400


@dataclass(frozen=True)
class Vehicle:
    x: float  # rear axle position (canonical no-slip bicycle reference point)
    y: float
    yaw: float
    v: float  # rear-axle forward speed
    steer: float  # front-wheel steering angle in radians, integrated


@dataclass
class Node:
    node_id: int
    layer: int
    state: Vehicle
    parent: Optional[int]
    action: Optional[tuple]  # (a_mps2, steering_rate_deg_s)
    station: float
    offset: float
    heading_error: float
    costs: tuple
    samples: list  # dense samples on incoming primitive (EXCLUDES source, INCLUDES destination)


def validate_config(cfg):
    for name in ('dt_s', 'integration_dt_s', 'speed_initial_mps', 'speed_min_mps',
                 'speed_max_mps', 'max_abs_steering_deg', 'wheelbase_m',
                 'max_lateral_accel_mps2', 'length_m', 'width_m',
                 'max_abs_route_offset_m', 'max_abs_heading_error_deg',
                 'obstacle_radius_m', 'clearance_margin_m', 'station_bin_m',
                 'lateral_bin_m', 'heading_bin_deg', 'speed_bin_mps', 'steering_bin_deg'):
        val = getattr(cfg, name)
        if not math.isfinite(val) or val <= 0:
            raise ValueError(name + ' must be finite and positive')
    if not 0 < cfg.max_abs_steering_deg < 90:
        raise ValueError('Steering bound must be less than 90 degrees')
    if not cfg.speed_min_mps <= cfg.speed_initial_mps <= cfg.speed_max_mps:
        raise ValueError('Initial speed must satisfy speed bounds')
    if cfg.horizon_steps < 1 or cfg.beam_width < 1:
        raise ValueError('Horizon/beam width must be positive')
    if cfg.road_area_tolerance_m2 < 0 or not math.isfinite(cfg.road_area_tolerance_m2):
        raise ValueError('Area tolerance must be finite and nonnegative')
    if not cfg.accelerations_mps2 or not cfg.steering_rates_deg_s:
        raise ValueError('Empty action set')
    if any(not math.isfinite(x) for x in (*cfg.accelerations_mps2, *cfg.steering_rates_deg_s)):
        raise ValueError('Nonfinite action')


def kinematic_derivative(q, a, omega, wheelbase):
    x, y, yaw, speed, steer = q
    return (speed * math.cos(yaw), speed * math.sin(yaw),
            speed * math.tan(steer) / wheelbase, a, omega)


def rk4_step(vehicle, a, steer_rate_deg_s, dt, wheelbase):
    """Exact continuity between primitives; numerical RK4 inside each primitive."""
    q = (vehicle.x, vehicle.y, vehicle.yaw, vehicle.v, vehicle.steer)
    omega = rad(steer_rate_deg_s)
    k1 = kinematic_derivative(q, a, omega, wheelbase)
    def plus(base, delta, factor):
        return tuple(v + factor * d for v, d in zip(base, delta))
    k2 = kinematic_derivative(plus(q, k1, dt/2), a, omega, wheelbase)
    k3 = kinematic_derivative(plus(q, k2, dt/2), a, omega, wheelbase)
    k4 = kinematic_derivative(plus(q, k3, dt), a, omega, wheelbase)
    result = tuple(qi + dt*(u+2*v+2*w+z)/6
                   for qi,u,v,w,z in zip(q,k1,k2,k3,k4))
    return Vehicle(result[0],result[1],wrap(result[2]),result[3],result[4])


def route_projection(route, x, y):
    """Nearest segment projection. Sign of offset is positive to LEFT of route."""
    best = None
    for a, b in zip(route, route[1:]):
        dx, dy = b[1]-a[1], b[2]-a[2]
        seg2 = dx*dx + dy*dy
        if seg2 < 1e-12:
            continue
        raw = ((x-a[1])*dx + (y-a[2])*dy)/seg2
        u = min(1.0, max(0.0, raw))
        px,py = a[1]+u*dx, a[2]+u*dy
        dist2 = (x-px)**2+(y-py)**2
        station = a[0]+u*(b[0]-a[0])
        offset = (dx*(y-a[2])-dy*(x-a[1]))/math.sqrt(seg2)
        yaw = math.atan2(dy,dx)
        if best is None or dist2 < best[0] - 1e-10 or (abs(dist2-best[0])<1e-10 and station > best[1]):
            best = (dist2,station,offset,yaw)
    if best is None:
        raise ValueError('Route has no valid segments')
    return best[1],best[2],best[3]


def initial_vehicle(route,cfg,start_station=0.0):
    if not 0 <= start_station < route[-1][0]:
        raise ValueError('Invalid start station')
    cx,cy,yaw = pose_at(route,start_station)
    # Assume geometric body center is halfway between front and rear axles.
    # Initial body center, not rear axle, coincides with the route start.
    return Vehicle(cx-0.5*cfg.wheelbase_m*math.cos(yaw),
                   cy-0.5*cfg.wheelbase_m*math.sin(yaw),
                   yaw,cfg.speed_initial_mps,0.0)


def body_center(v,cfg):
    # Simplified geometry: rectangle center halfway along wheelbase from rear axle.
    return (v.x+0.5*cfg.wheelbase_m*math.cos(v.yaw),
            v.y+0.5*cfg.wheelbase_m*math.sin(v.yaw))


def _footprint(v,cfg):
    cx,cy=body_center(v,cfg)
    return vehicle_rectangle(cx,cy,v.yaw,cfg.length_m,cfg.width_m)


def footprint_outside_area(v,road,cfg,prepared=None):
    f=_footprint(v,cfg)
    prepared = prepared if prepared is not None else prep(road)
    if prepared.covers(f):
        return 0.0
    return f.difference(road).area


def primitive(parent, accel, steer_rate, cfg, route, road, obstacle, prepared=None):
    """Return (samples, edge costs, station, offset, heading error) or reason.

    Samples each <= integration_dt_s, including endpoint. Checks actual
    integrated rectangle against road & circular obstacle at every sample.
    R9 cost is soft obstacle CLEARANCE SHORTFALL of the rectangle boundary;
    R5 cost integrated squared signed centerline offset; R4 is action effort.
    """
    prepared = prepared or prep(road)
    observation = Point(obstacle)
    current=parent.state
    if current.steer + rad(steer_rate)*cfg.dt_s > rad(cfg.max_abs_steering_deg)+1e-9 or current.steer + rad(steer_rate)*cfg.dt_s < -rad(cfg.max_abs_steering_deg)-1e-9:
        return None,'steering_limit'
    n=max(2,math.ceil(cfg.dt_s/cfg.integration_dt_s))
    h=cfg.dt_s/n
    samples=[]
    r9=r5=0.0
    # midpoint/trapezoid integration of sampled cost at start and endpoints.
    def evaluate(v):
        if v.v < cfg.speed_min_mps-1e-7 or v.v > cfg.speed_max_mps+1e-7:
            return None,'speed_limit'
        if abs(v.steer) > rad(cfg.max_abs_steering_deg)+1e-7:
            return None,'steering_limit'
        lat_a=v.v*v.v*abs(math.tan(v.steer))/cfg.wheelbase_m
        if lat_a > cfg.max_lateral_accel_mps2+1e-7:
            return None,'lateral_accel_limit'
        cx,cy=body_center(v,cfg)
        s,d,tangent=route_projection(route,cx,cy)
        heading=wrap(v.yaw-tangent)
        if abs(d) > cfg.max_abs_route_offset_m+1e-7 or abs(heading) > rad(cfg.max_abs_heading_error_deg)+1e-7:
            return None,'route_deviation_limit'
        f=_footprint(v,cfg)
        if not prepared.covers(f) and f.difference(road).area > cfg.road_area_tolerance_m2:
            return None,'road_footprint'
        clearance=f.distance(observation)-cfg.obstacle_radius_m
        if clearance < -1e-7:
            return None,'hard_obstacle_collision'
        return (s,d,heading,max(0.0,cfg.clearance_margin_m-clearance),d*d),None
    prev,reason=evaluate(current)
    if reason:
        return None,'invalid_source_'+reason
    for j in range(1,n+1):
        current=rk4_step(current,accel,steer_rate,h,cfg.wheelbase_m)
        cur,reason=evaluate(current)
        if reason:
            return None,reason
        r9+=h*(prev[3]+cur[3])*0.5
        r5+=h*(prev[4]+cur[4])*0.5
        samples.append((j*h,current,cur[0],cur[1],cur[2]))
        prev=cur
    r4=cfg.dt_s*(accel*accel+(steer_rate/8.0)**2)
    return (samples,(r9,r5,r4),prev[0],prev[1],prev[2]),None


def _bucket(n,cfg):
    return (round(n.station/cfg.station_bin_m),
            round(n.offset/cfg.lateral_bin_m),
            round(math.degrees(n.heading_error)/cfg.heading_bin_deg),
            round(n.state.v/cfg.speed_bin_mps),
            round(math.degrees(n.state.steer)/cfg.steering_bin_deg))


def select_diverse_candidates(candidates,cfg):
    """Rulebook-agnostic approximate beam pruning, no re-quantization of pose.

    One closest-to-bin-center representative per feature bin, then fair
    round-robin over station/offset groups if the unique-bin pool exceeds beam.
    This is a limited search, NOT complete or Pareto-optimal motion planning.
    """
    winners={}
    for node in candidates:
        key=_bucket(node,cfg)
        residual=(abs(node.station/cfg.station_bin_m-key[0])+abs(node.offset/cfg.lateral_bin_m-key[1])+
                  abs(math.degrees(node.heading_error)/cfg.heading_bin_deg-key[2])+
                  abs(node.state.v/cfg.speed_bin_mps-key[3])+
                  abs(math.degrees(node.state.steer)/cfg.steering_bin_deg-key[4]))
        if key not in winners or (residual,node.node_id)<winners[key][0]:
            winners[key]=((residual,node.node_id),node)
    unique=[item[1] for item in winners.values()]
    if len(unique)<=cfg.beam_width:
        return sorted(unique,key=lambda n:n.node_id)
    groups=defaultdict(list)
    for node in unique:
        key=(round(node.station/4.0),round(node.offset/0.5))
        groups[key].append(node)
    for nodes in groups.values():
        nodes.sort(key=lambda n:(abs(n.heading_error),abs(n.state.steer),n.node_id))
    # Alternate station/lateral groups in stable order; independent of R9/R5.
    keys=sorted(groups,key=lambda z:(z[0],z[1]))
    chosen=[]
    while keys and len(chosen)<cfg.beam_width:
        next_keys=[]
        for key in keys:
            if len(chosen)==cfg.beam_width:
                break
            chosen.append(groups[key].pop(0))
            if groups[key]:
                next_keys.append(key)
        keys=next_keys
    return sorted(chosen,key=lambda n:n.node_id)


def build_motion_tree(route,road,cfg,start_station=0.0):
    validate_config(cfg)
    if road.is_empty or not road.is_valid:
        raise ValueError('Road geometry invalid; do not silently repair map')
    if cfg.obstacle_station_m>=route[-1][0] or cfg.obstacle_station_m<0:
        raise ValueError('Obstacle outside route')
    start=initial_vehicle(route,cfg,start_station)
    if footprint_outside_area(start,road,cfg)>cfg.road_area_tolerance_m2:
        raise ValueError('Initial vehicle footprint outside road')
    obs=world_xy(route,cfg.obstacle_station_m,cfg.obstacle_offset_m)
    prepared=prep(road)
    cx,cy=body_center(start,cfg)
    s,d,tangent=route_projection(route,cx,cy)
    root=Node(0,0,start,None,None,s,d,wrap(start.yaw-tangent),(0.,0.,0.),[])
    nodes={0:root}; layer=[root]; layers=[]; reject=Counter(); next_id=1
    for step in range(cfg.horizon_steps):
        candidates=[]
        for src in layer:
            for a in cfg.accelerations_mps2:
                for omega in cfg.steering_rates_deg_s:
                    result,reason=primitive(src,a,omega,cfg,route,road,obs,prepared)
                    if reason:
                        reject[reason]+=1
                        continue
                    samples,cost,sn,dn,psi=result
                    end=samples[-1][1]
                    total=tuple(src.costs[i]+cost[i] for i in range(3))
                    candidates.append(Node(next_id,step+1,end,src.node_id,(a,omega),sn,dn,psi,total,samples))
                    next_id+=1
        if not candidates:
            raise RuntimeError('No feasible bicycle primitives at step {} (rejections: {})'.format(step+1,dict(reject)))
        chosen=select_diverse_candidates(candidates,cfg)
        for node in chosen:
            nodes[node.node_id]=node
        layers.append({'step':step+1,'parents':len(layer),'viable_candidate_edges':len(candidates),
                       'retained_states':len(chosen),'pruned_candidates':len(candidates)-len(chosen)})
        layer=chosen
    return nodes,layer,layers,dict(reject),obs


def rank(node,profile,min_final_s):
    c=node.costs
    # Preference: R9/R5, progress, comfort. Never use comfort before progress.
    if node.station < min_final_s-1e-8:
        return None
    if profile=='safety_first':
        return (round(c[0],9),round(c[1],9),-node.station,round(c[2],9),node.node_id)
    if profile=='corridor_first':
        return (round(c[1],9),round(c[0],9),-node.station,round(c[2],9),node.node_id)
    raise ValueError('Unknown priority')


def select_path(nodes,terminal_nodes,profile,min_final_s=0):
    eligible=[(rank(n,profile,min_final_s),n) for n in terminal_nodes]
    eligible=[item for item in eligible if item[0] is not None]
    if not eligible:
        raise RuntimeError('No full-horizon terminal state meets min-final-s={:.1f}'.format(min_final_s))
    final=min(eligible,key=lambda p:p[0])[1]
    path=[final]
    while path[-1].parent is not None:
        path.append(nodes[path[-1].parent])
    path.reverse()
    return path


def path_samples(path,cfg,profile,route):
    rows=[]
    root=path[0]
    rows.append(_sample_row(root.state,cfg,0.,root.station,root.offset,root.heading_error,profile,0.,0.))
    for i,node in enumerate(path[1:]):
        a,omega=node.action
        for tau,state,s,d,heading in node.samples:
            rows.append(_sample_row(state,cfg,i*cfg.dt_s+tau,s,d,heading,profile,a,omega))
    return rows


def _sample_row(q,cfg,t,s,d,heading,profile,a,omega):
    cx,cy=body_center(q,cfg)
    return {'profile':profile,'t_s':t,'x_m':cx,'y_m':cy,
            'rear_axle_x_m':q.x,'rear_axle_y_m':q.y,
            'yaw_rad':q.yaw,'speed_mps':q.v,'steering_deg':math.degrees(q.steer),
            'station_m':s,'offset_m':d,'heading_error_deg':math.degrees(heading),
            'acceleration_mps2':a,'steering_rate_deg_s':omega}


def compare_paths(a,b):
    if len(a)!=len(b):
        raise ValueError('Different dense sample count')
    # Identical times and steps by construction, verify rather than assume.
    if any(abs(x['t_s']-y['t_s'])>1e-7 for x,y in zip(a,b)):
        raise ValueError('Sample times do not align')
    ds=[math.hypot(x['x_m']-y['x_m'],x['y_m']-y['y_m']) for x,y in zip(a,b)]
    return {'max_time_aligned_separation_m':max(ds),
            'final_separation_m':ds[-1],
            'max_station_difference_m':max(abs(x['station_m']-y['station_m']) for x,y in zip(a,b)),
            'max_lateral_offset_difference_m':max(abs(x['offset_m']-y['offset_m']) for x,y in zip(a,b)),
            'final_station_difference_m':abs(a[-1]['station_m']-b[-1]['station_m'])}


def check_selected(path,road,obstacle,cfg):
    point=Point(obstacle)
    all_q=[path[0].state]+[sample[1] for n in path[1:] for sample in n.samples]
    outside=max(footprint_outside_area(q,road,cfg) for q in all_q)
    min_obstacle_clearance=min(_footprint(q,cfg).distance(point)-cfg.obstacle_radius_m for q in all_q)
    return {'sample_count':len(all_q),'maximum_road_outside_area_m2':outside,
            'minimum_obstacle_surface_clearance_m':min_obstacle_clearance,
            'road_sampled_pass':outside<=cfg.road_area_tolerance_m2+1e-8,
            'hard_obstacle_sampled_pass':min_obstacle_clearance>=-1e-7,
            'max_abs_steering_deg':max(abs(math.degrees(q.steer)) for q in all_q),
            'max_lateral_accel_mps2':max(q.v*q.v*abs(math.tan(q.steer))/cfg.wheelbase_m for q in all_q),
            'max_integration_endpoint_mismatch_m':max((math.hypot(node.samples[-1][1].x-node.state.x,
                        node.samples[-1][1].y-node.state.y)
                    for node in path[1:]),default=0.0)}


def save_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def _paint_polygon(ax,shape):
    if shape.geom_type=='Polygon':
        x,y=shape.exterior.xy
        ax.fill(x,y,color='0.7',alpha=0.35,zorder=0)
    elif shape.geom_type in ('MultiPolygon','GeometryCollection'):
        for item in shape.geoms:
            _paint_polygon(ax,item)


def plot_comparison(path,route,road,obstacle,sa,sb,cfg):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except ImportError:
        return False
    fig,ax=plt.subplots(figsize=(10,5))
    xmin=min(min(r['x_m'] for r in sa),min(r['x_m'] for r in sb),obstacle[0])-5
    xmax=max(max(r['x_m'] for r in sa),max(r['x_m'] for r in sb),obstacle[0])+5
    ymin=min(min(r['y_m'] for r in sa),min(r['y_m'] for r in sb),obstacle[1])-4
    ymax=max(max(r['y_m'] for r in sa),max(r['y_m'] for r in sb),obstacle[1])+4
    _paint_polygon(ax,road.intersection(box(xmin,ymin,xmax,ymax)))
    ax.plot([p[1] for p in route],[p[2] for p in route],':',color='0.5',lw=1,label='Reference route')
    ax.plot([r['x_m'] for r in sa],[r['y_m'] for r in sa],label='R9 first (bicycle)')
    ax.plot([r['x_m'] for r in sb],[r['y_m'] for r in sb],label='R5 first (bicycle)')
    ax.add_patch(Circle(obstacle,cfg.obstacle_radius_m,alpha=0.5,color='orange',label='Synthetic obstacle'))
    ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax)
    ax.set_aspect('equal',adjustable='box');ax.set_xlabel('CommonRoad X (m)');ax.set_ylabel('CommonRoad Y (m)')
    ax.set_title('Milestone 4E-B: integrated bicycle-model motion paths (offline)')
    ax.grid(alpha=0.15);ax.legend(fontsize=8,loc='best')
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)
    return True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--route',type=Path,default=DEFAULT_ROUTE)
    p.add_argument('--map',type=Path,default=DEFAULT_MAP)
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--demo',action='store_true',help='Synthetic westbound road; does not use Town01 XML')
    p.add_argument('--beam-width',type=int,default=Config.beam_width)
    p.add_argument('--min-final-s',type=float,default=0)
    p.add_argument('--start-s',type=float,default=0)
    p.add_argument('--sample-dt',type=float,default=Config.integration_dt_s)
    p.add_argument('--obstacle-s',type=float,default=Config.obstacle_station_m)
    p.add_argument('--obstacle-d',type=float,default=Config.obstacle_offset_m)
    args=p.parse_args()
    from dataclasses import replace
    cfg=replace(Config(),beam_width=args.beam_width,integration_dt_s=args.sample_dt,
                obstacle_station_m=args.obstacle_s,obstacle_offset_m=args.obstacle_d)
    validate_config(cfg)
    if not math.isfinite(args.min_final_s) or args.min_final_s<0:
        p.error('min-final-s must be finite and nonnegative')
    if args.demo:
        route=synthetic_route()
        road=box(-4.,-1.75,84.,1.75)
        lanelet_count=None
    else:
        if not args.route.is_file() or not args.map.is_file():
            p.error('Missing Town01 route CSV or map XML. Run earlier milestones first.')
        route=read_route_csv(args.route)
        road,lanelet_count=load_town01_road(args.map,VALIDATION_LANELETS)
    nodes,terminals,layers,rejections,obs=build_motion_tree(route,road,cfg,args.start_s)
    selected={}
    for profile in ('safety_first','corridor_first'):
        path=select_path(nodes,terminals,profile,args.min_final_s)
        dense=path_samples(path,cfg,profile,route)
        diagnostic=check_selected(path,road,obs,cfg)
        if not diagnostic['road_sampled_pass'] or not diagnostic['hard_obstacle_sampled_pass']:
            raise AssertionError('Internal error: a selected path failed checks')
        selected[profile]={'path':path,'dense':dense,'diagnostic':diagnostic}
    div=compare_paths(selected['safety_first']['dense'],selected['corridor_first']['dense'])
    args.out.mkdir(parents=True,exist_ok=True)
    for profile,data in selected.items():
        save_csv(args.out/(profile+'_bicycle_samples.csv'),data['dense'])
    plotted=plot_comparison(args.out/'comparison.png',route,road,obs,
                            selected['safety_first']['dense'],selected['corridor_first']['dense'],cfg)
    report={
        'experiment':'Milestone 4E-B: beam-pruned exact-state bicycle motion tree with priority proxy selection',
        'route_source':'synthetic-demo' if args.demo else str(args.route),
        'route_sha256':None if args.demo else sha(args.route),
        'map_source':'synthetic-3.5m-road' if args.demo else str(args.map),
        'map_sha256':None if args.demo else sha(args.map),
        'lanelets_validation':None if args.demo else list(VALIDATION_LANELETS),
        'lanelets_planned':None if args.demo else list(PLANNED_LANELETS),
        'map_lanelet_count':lanelet_count,
        'obstacle_xy_m':list(obs), 'settings':asdict(cfg),'start_station_m':args.start_s,
        'min_final_station_m':args.min_final_s,
        'shared_graph':True,
        'search_is_bounded_approximation':True,
        'tree_nodes_retained':len(nodes),
        'tree_edges_retained':len(nodes)-1,
        'viable_candidates_before_beam':sum(x['viable_candidate_edges'] for x in layers),
        'beam_pruned_candidates':sum(x['pruned_candidates'] for x in layers),
        'layers':layers,'rejections_by_reason':rejections,
        'profiles':{name:{'costs':{'R9_footprint_clearance_shortfall_proxy':data['path'][-1].costs[0],
                                   'R5_centerline_offset_squared_proxy':data['path'][-1].costs[1],
                                   'R4_action_effort_proxy':data['path'][-1].costs[2]},
                          'final_station_m':data['path'][-1].station,
                          'diagnostic':data['diagnostic'],
                          'actions':[{'acceleration_mps2':n.action[0], 'steering_rate_deg_s':n.action[1]}
                                     for n in data['path'][1:]]}
                    for name,data in selected.items()},
        'divergence':div,'plot_written':plotted,
        'limits_and_warnings':[
            'AI-authored educational motion-tree prototype, NOT upstream CommonRoad Rulebook implementation.',
            'R9 uses rectangle-to-point clearance shortfall; R5 integrates centerline offset squared; these differ from earlier M4B cost formulations.',
            'Lexicographic TOTAL priority orders, not partially ordered Rulebook dominance.',
            'Motion-tree is beam-pruned using priority-independent geometric bins. Results not globally optimal.',
            'Steering-rate control is explicitly integrated; simple kinematic bicycle (no tire forces, throttle/steer actuator dynamics).',
            'Road and obstacle collision sampled at integration_dt_s, not continuously certified.',
            'Static circular synthetic obstacle, rectangle ego footprint; not CARLA actors.',
            'Bicycle state follows rear axle; rectangle center assumed wheelbase/2 ahead of rear axle, not calibrated to actual CARLA geometry.',
            'Only selected lanelet road union; no legal lane changes, signage, red lights, dynamic obstacles, or traffic rules.',
            'No CARLA control, feedback tracking, or Scenic generation.',
        ]}
    (args.out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('\n===== MILESTONE 4E-B: BICYCLE-MODEL MOTION TREE =====')
    print('Source:',report['route_source'])
    print('Retained states/edges:',report['tree_nodes_retained'],report['tree_edges_retained'])
    print('Viable candidate attempts:',report['viable_candidates_before_beam'],
          'beam-pruned:',report['beam_pruned_candidates'])
    print('Rejection reasons:',report['rejections_by_reason'])
    for name,data in selected.items():
        q=data['path'][-1]
        print('{}: R9={:.4f}, R5={:.4f}, R4={:.3f}, final s={:.2f} m, road={}, collision={}, peak steer={:.2f} deg'.format(
              name,*q.costs,q.station,data['diagnostic']['road_sampled_pass'],
              data['diagnostic']['hard_obstacle_sampled_pass'],data['diagnostic']['max_abs_steering_deg']))
    print('Max aligned separation={:.3f}m, max lateral difference={:.3f}m, final progress difference={:.3f}m'.format(
          div['max_time_aligned_separation_m'],div['max_lateral_offset_difference_m'],div['final_station_difference_m']))
    print('Outputs:',args.out)
    print('LIMIT: bounded offline search, sampled checks, proxy rules; NOT CARLA-ready.')


if __name__=='__main__':
    main()
