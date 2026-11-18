"""M6B-5 offline: route-derived, obstacle-free, low-speed Town01 reference.

Uses *existing CommonRoad-converted route CSV* and 4E-B map. This is a new
route-based reference, NOT an upstream Rulebook solution, NOT an empirical CARLA
recording, and NOT a CARLA control program.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

# Calibration supplies stable-observation loading without importing CARLA here.
import carla_calibrate_m6b3 as b
# The model module retains the public milestone version; common primitives are
# imported from their shared implementation to prevent duplicate definitions.
from carla_model_m6b4 import VERSION
from carla_m6b_common import DT, digest, finite, wrap

HERE=Path(__file__).resolve().parent
DEFAULT_OBS=HERE/'output/carla_observation_6b2_stable/trial_003/live_observation.json'
DEFAULT_MODEL=HERE/'output/carla_calibration_6b4/model_v1.json'
DEFAULT_ROUTE=HERE/'output/town01_graph/route_waypoints.csv'
DEFAULT_MAP=HERE/'maps/Town01.rpp_compat.xml'
DEFAULT_SUMMARY=HERE/'output/town01_bicycle_graph_beam800_min40/summary.json'
DEFAULT_OUT=HERE/'output/carla_calibration_6b5/route_reference_v1.json'
SCHEMA='m6b5-commonroad-route-reference-v1'
COUNT=85
CAP_SPEED=1.65
CAP_COMMAND=.08
# These are PROVISIONAL reference velocities, not a CARLA drivetrain fit.
# Low-speed throttle schedule is separately checked against known 6B-3 controls.

def nominal_speed(t):
    t=finite('reference time',t)
    if not 0<=t<=4.2+1e-7:raise ValueError('Reference time outside tested 4.2s window')
    if t<=1.8:return .9*t
    if t<=2.2:return 1.62-.25*(t-1.8)
    return max(0.,1.52-12.*(t-2.2))


def actuator_schedule(index):
    """Use previously exercised throttle/brake magnitudes, not recorded path samples."""
    if index==0:return 'initial',0.,0.
    if not 1<=index<COUNT:raise ValueError('Invalid index')
    t=(index-1)*DT
    if t<1.1-1e-9:return 'throttle_straight',.25,0.
    if t<1.8-1e-9:return 'route_follow_throttle',.20,0.
    if t<2.2-1e-9:return 'coast',0.,0.
    return 'brake',0.,.65


def validated_model(path,observation_hash):
    model=json.loads(Path(path).read_text(encoding='utf-8'))
    if (model.get('schema_version')!=VERSION or model.get('status')!='provisional_offline_fit' or
        model.get('observation_sha256')!=observation_hash or
        model.get('manual_gear')!=1 or model.get('automatic_mode_validated') is not False or
        model.get('carla_command_to_commonroad_steering_sign')!=-1 or
        model.get('carla_map')!='Carla/Maps/Town01'):
        raise ValueError('Calibrated model inconsistent with stable observed CARLA source')
    wheelbase=finite('wheelbase',model['geometric_wheelbase_m'])
    gain=finite('front wheel gain',model['front_wheel_signed_deg_per_normalized_steer'])
    if not 2.5<wheelbase<3.5 or not 40<gain<90 or finite('maximum calibrated steering',model['calibrated_steer_command_abs_max'])<.08-1e-8:
        raise ValueError('Calibration parameters exceed validated range')
    return model


def check_4eb_provenance(summary,route_path,map_path):
    if (not str(summary.get('experiment','')).startswith('Milestone 4E-B') or
        summary.get('route_sha256')!=digest(route_path) or
        summary.get('map_sha256')!=digest(map_path)):
        raise ValueError('CommonRoad route/map do not match the recorded 4E-B source')


def project_geometry(route,road,obs,model,pose_at,project,rectangle):
    """Build 85 timed samples using ONLY route geometry and a declared speed hypothesis."""
    x0,y0=map(float,obs['body_center_cr_xy_m'])
    yaw0=finite('stable yaw',obs['yaw_cr_rad'])
    s0,d0,psi0=project(route,x0,y0)
    if not(math.isfinite(s0) and abs(d0)<=.02 and abs(math.degrees(wrap(yaw0-psi0)))<=.5):
        raise ValueError('CommonRoad route origin does not align with stable CARLA start')
    route_x0,route_y0,route_yaw0=pose_at(route,s0)
    if (math.hypot(route_x0-x0,route_y0-y0)>.02 or
        abs(math.degrees(wrap(route_yaw0-yaw0)))>.5):
        raise ValueError('Projected route reference start does not match stable CARLA body-center pose')
    if s0<0 or s0+3.0>=route[-1][0]:
        raise ValueError('Insufficient verified route length beyond start')
    gain_rad=math.radians(float(model['front_wheel_signed_deg_per_normalized_steer']))
    wheelbase=float(model['geometric_wheelbase_m'])
    length=4.791779518127441;width=2.163450002670288  # measured 6B-2S Model 3 bbox
    samples=[];s=s0;prev_speed=0.;prev_s=s0
    max_outside=0.
    for i in range(COUNT):
        t=i*DT;v=nominal_speed(t)
        if i:s+=DT*(prev_speed+v)/2
        if s>=route[-1][0]:raise ValueError('Reference reaches route end')
        x,y,yaw=pose_at(route,s)
        # Curvature from route tangent change over a finite station window.
        sa=max(0.,s-.30);sb=min(route[-1][0]-.000001,s+.30)
        ya=pose_at(route,sa)[2];yb=pose_at(route,sb)[2]
        kappa=wrap(yb-ya)/(sb-sa)
        steering=math.atan(wheelbase*kappa)/(-gain_rad)
        if abs(steering)>CAP_COMMAND+1e-8:
            raise ValueError(f'Route curvature requires unvalidated steering at t={t:.2f}')
        if abs(kappa)>0.035:
            raise ValueError('Route curvature outside low-speed straight-first reference envelope')
        if abs(math.degrees(wrap(yaw-yaw0)))>4.0:
            raise ValueError('Route changes heading beyond validated low-speed envelope')
        # Sampled geometry check on actual 6B-2S vehicle body rectangle.
        footprint=rectangle(x,y,yaw,length,width)
        outside=footprint.difference(road).area
        max_outside=max(max_outside,outside)
        if outside>1e-5:raise ValueError(f'Route body footprint leaves CommonRoad road polygon at t={t:.2f}')
        phase,throttle,brake=actuator_schedule(i)
        samples.append({'index':i,'t_s':round(t,5), 'station_m':s,
            'x_cr_m':x,'y_cr_m':y,'yaw_cr_rad':wrap(yaw),
            'reference_speed_mps':v, 'reference_acceleration_mps2':0. if i==0 else (v-prev_speed)/DT,
            'route_curvature_inv_m':kappa,'feedforward_steer_normalized':steering,
            'phase':phase,'command_throttle':throttle,'command_brake':brake,
            'reference_lateral_offset_m':0.})
        prev_speed=v;prev_s=s
    if abs(s-s0-2.186)>0.08 or not 0. < s-s0 < 3.0 or samples[-1]['reference_speed_mps']!=0:
        raise ValueError('Unexpected reference distance or nonstationary endpoint')
    return samples,{'start_station_m':s0,'start_offset_m':d0,
        'reference_displacement_station_m':s-s0,'maximum_sampled_outside_road_area_m2':max_outside,
        'max_reference_speed_mps':max(r['reference_speed_mps'] for r in samples),
        'max_feedforward_abs_steer':max(abs(r['feedforward_steer_normalized']) for r in samples),
        'max_curvature_inv_m':max(abs(r['route_curvature_inv_m']) for r in samples)}


def geometry_from_project(route_path,map_path):
    """Use the project's original CommonRoad-conversion route and lanelet utilities."""
    from town01_motion_graph import read_route_csv,pose_at
    from town01_bicycle_graph import route_projection,VALIDATION_LANELETS
    from town01_feasibility_check import load_town01_road,vehicle_rectangle
    route=read_route_csv(route_path)
    road,n=load_town01_road(map_path,VALIDATION_LANELETS)
    if len(route)<2 or not road.is_valid or road.is_empty or int(n)!=300:
        raise ValueError('Unexpected/invalid CommonRoad Town01 road geometry')
    return route,road,n,pose_at,route_projection,vehicle_rectangle


def generate(obs,model,model_path,route_path,map_path,summary_path,geometry=None):
    summary=json.loads(Path(summary_path).read_text(encoding='utf-8'))
    check_4eb_provenance(summary,route_path,map_path)
    if geometry is None:geometry=geometry_from_project(route_path,map_path)
    route,road,lanelets,pose_at,project,rectangle=geometry
    rows,audit=project_geometry(route,road,obs,model,pose_at,project,rectangle)
    return {'schema_version':SCHEMA,'status':'offline_reference_generated','milestone':'6B-5',
        'reference_source':'CommonRoad-converted Town01 4E-B route CSV and road geometry; NOT empirical CARLA trajectory',
        'original_rulebook_planner_used':False,'scenic_used':False,'obstacle_included':False,
        'dt_s':DT,'count':COUNT,'source_observation_sha256':obs['sha256'],
        'source_observation_frame':obs['source_frame'],
        'calibrated_model_sha256':digest(model_path),
        'route_sha256':digest(route_path),'map_sha256':digest(map_path),
        '4eb_summary_sha256':digest(summary_path),
        'validation_lanelet_count':lanelets,
        'reference_origin_cr':{'x_cr_m':rows[0]['x_cr_m'],'y_cr_m':rows[0]['y_cr_m'],'yaw_cr_rad':rows[0]['yaw_cr_rad']},
        'physics_model':{'wheelbase_m':float(model['geometric_wheelbase_m']),
                        'carla_to_cr_steer_sign':-1,
                        'front_wheel_gain_deg_per_normalized':float(model['front_wheel_signed_deg_per_normalized_steer'])},
        'longitudinal_reference_hypothesis':{
            'description':'Independent scripted speed curve: 0.90m/s2 ramp to 1.8s, mild coast decay, fast stop after 2.2s',
            'not_a_fitted_engine_or_brake_model':True,
            'nominal_accel_mps2':.9,'nominal_coast_decel_mps2':.25,'nominal_brake_decel_mps2':12.,
            'controls':'Throttle/brake values previously exercised in 6B-3, not derived from recorded CARLA path'},
        'offline_feasibility':audit,'samples':rows,
        'limitations':[
            'Route derived from CommonRoad-converted Town01 planning artifact, not a new Rulebook optimization',
            'No obstacle and no Scenic scenario; reference excludes all previous CARLA trajectory samples',
            'Nominal speed is a hypothesis, not calibrated acceleration/braking dynamics',
            'Road and vehicle-box checks are sampled and geometric; not continuous collision certification',
            'Static prop exclusion and other actors must be rechecked before and during any driving',
            'Wheel angle gain and geometric wheelbase are provisional and validated only at low speed',
        ]}


def load_reference(path,obs,model,model_path,route_path,map_path,summary_path,geometry=None):
    actual=json.loads(Path(path).read_text(encoding='utf-8'))
    if actual.get('schema_version')!=SCHEMA or actual.get('status')!='offline_reference_generated':
        raise ValueError('Not a completed M6B-5 route reference')
    expected=generate(obs,model,model_path,route_path,map_path,summary_path,geometry)
    if actual!=expected:raise ValueError('CommonRoad reference differs from regenerated source geometry/model')
    return actual


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observation',type=Path,default=DEFAULT_OBS)
    p.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    p.add_argument('--route',type=Path,default=DEFAULT_ROUTE)
    p.add_argument('--map',type=Path,default=DEFAULT_MAP)
    p.add_argument('--source-summary','--4eb-summary',dest='source_summary',type=Path,default=DEFAULT_SUMMARY)
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    a=p.parse_args(argv)
    try:
        if a.out.exists():raise FileExistsError('Refusing to overwrite an existing route reference')
        obs=b.read_stable_observation(a.observation)
        model=validated_model(a.model,obs['sha256'])
        result=generate(obs,model,a.model,a.route,a.map,a.source_summary)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print('M6B-5 ROUTE REFERENCE:',result['status'])
        print('Progress:',result['offline_feasibility']['reference_displacement_station_m'],'m')
        print('Route and map SHA-256 verified; sampled vehicle footprint inside road')
        print('NO CARLA connection/control, no obstacle, no Rulebook/Scenic. Output:',a.out)
        return 0
    except (ValueError,RuntimeError,KeyError,TypeError,ImportError,OSError) as e:
        print('M6B-5 ROUTE REFERENCE FAILED SAFE:',type(e).__name__+': '+str(e));return 2

if __name__=='__main__':raise SystemExit(main())
