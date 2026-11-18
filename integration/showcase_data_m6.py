"""Offline CARLA integration showcase source loader. Zero simulator dependency.

Every visible path is loaded from a real experiment CSV. This module NEVER
synthesizes recorded CARLA trajectories or equates offline plans with driven motion.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import math
from pathlib import Path

DT = 0.05
ROOT_RELATIVE_PATHS = {
    'observation': 'integration/output/carla_observation_6b2_stable/trial_003/live_observation.json',
    'plan_report': 'integration/output/carla_plan_6b2/obstacle_encounter_001/report.json',
    'plan_safety': 'integration/output/carla_plan_6b2/obstacle_encounter_001/safety_first_observed_samples.csv',
    'plan_corridor': 'integration/output/carla_plan_6b2/obstacle_encounter_001/corridor_first_observed_samples.csv',
    'straight': 'integration/output/carla_calibration_6b3/manual_first_001/calibration_samples.csv',
    'steer_pos': 'integration/output/carla_calibration_6b3/manual_steering_pos_001/calibration_samples.csv',
    'steer_neg': 'integration/output/carla_calibration_6b3/manual_steering_neg_001/calibration_samples.csv',
    'empirical_off': 'integration/output/carla_calibration_6b4b/feedback_off_001/calibration_samples.csv',
    'empirical_on': 'integration/output/carla_calibration_6b4b/feedback_on_001/calibration_samples.csv',
    'route_off': 'integration/output/carla_calibration_6b5/route_off_001/calibration_samples.csv',
    'route_on': 'integration/output/carla_calibration_6b5/route_on_001/calibration_samples.csv',
    'speed_failed': 'integration/output/carla_calibration_6b6/speed_on_001/calibration_samples.csv',
    'speed_fixed': 'integration/output/carla_calibration_6b6/speed_on_002_coastfix/calibration_samples.csv',
}

SCENE_KEYS = {
    'intro': [],
    'observation': ['observation'],
    'planning': ['observation','plan_report','plan_safety','plan_corridor'],
    'straight': ['straight'],
    'steer_pos': ['steer_pos'],
    'steer_neg': ['steer_neg'],
    'feedback': ['empirical_off','empirical_on'],
    'route_off': ['route_off'],
    'route_on': ['route_on'],
    'speed_failed': ['speed_failed'],
    'speed_fixed': ['speed_fixed'],
    'outro': [],
}
SCENE_ORDER = tuple(SCENE_KEYS)
ARCHIVE_SCENES = {'straight','steer_pos','steer_neg','feedback','route_off','route_on','speed_failed','speed_fixed'}
PLANNING_SCENES = {'planning'}

@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def number(v, name):
    try: x=float(v)
    except (TypeError, ValueError) as exc: raise ValueError(f'Invalid {name}: {v!r}') from exc
    if not math.isfinite(x): raise ValueError(f'Non-finite {name}')
    return x

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def read_samples(path, *, planning=False):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    if not (2 <= len(rows) <= 500):
        raise ValueError(f'{path}: expected 2..500 samples, got {len(rows)}')
    keys=['x_m','y_m','yaw_rad','t_s'] if planning else ['x_cr_m','y_cr_m','yaw_cr_rad','t_s']
    for i,row in enumerate(rows):
        for k in keys: number(row.get(k),f'{path}:{i}:{k}')
        t=number(row['t_s'],'time')
        if i and not number(rows[i-1]['t_s'],'previous time') < t:
            raise ValueError(f'{path}: time is not increasing at row {i}')
        if not planning and i>0 and abs(t - i*DT) > .001:
            raise ValueError(f'{path}: expected {DT}s timestep at row {i}')
    return rows

def pose(row, *, planning=False):
    x,y,yaw=('x_m','y_m','yaw_rad') if planning else ('x_cr_m','y_cr_m','yaw_cr_rad')
    return Pose(number(row[x],x),number(row[y],y),number(row[yaw],yaw))

def observation_pose(doc):
    if doc.get('stability_capture_mode') != 'passive_multi_frame' or doc.get('stability',{}).get('latest',{}).get('accepted') is not True:
        raise ValueError('Source observation must be verified settled capture')
    if doc.get('cleanup_ok') is not True:
        raise ValueError('Source capture did not pass cleanup')
    ego=doc['ego']
    cx,cy=ego['body_center_cr_xy_m']
    return Pose(number(cx,'ego x'),number(cy,'ego y'),number(ego['yaw_cr_rad'],'ego yaw'))

def check_plan_report(doc):
    if doc.get('status') != 'offline_plan_generated' or doc.get('executed_carla_control') is not False:
        raise ValueError('Planner report is not validated offline-only planning')
    d=doc.get('divergence',{})
    if number(d.get('max_time_aligned_separation_m'), 'divergence') <= .1:
        raise ValueError('Expected nonzero trajectory divergence')
    return d

def validate_sources(root, scenes=None, *, skip_missing=False):
    root=Path(root).resolve()
    scenes = tuple(scenes or SCENE_ORDER)
    unknown=set(scenes)-set(SCENE_KEYS)
    if unknown: raise ValueError(f'Unknown showcase scenes: {sorted(unknown)}')
    required=set(k for scene in scenes for k in SCENE_KEYS[scene])
    required.add('observation') # every scene needs a grounded, settled Town01 starting pose
    paths={k:root/Path(rel) for k,rel in ROOT_RELATIVE_PATHS.items() if k in required}
    missing=[str(paths[k]) for k in sorted(required) if not paths[k].is_file()]
    if (not paths['observation'].is_file()) or (missing and not skip_missing):
        raise FileNotFoundError('Showcase source data missing:\n'+'\n'.join(missing))
    included=[]
    for s in scenes:
        if all(paths[k].is_file() for k in SCENE_KEYS[s]): included.append(s)
    if not included: raise ValueError('No valid scenes found')
    data={}
    for k in set(['observation']+[key for s in included for key in SCENE_KEYS[s]]):
        p=paths[k]
        data[k]=read_json(p) if k.endswith('report') or k=='observation' else read_samples(p,planning=k.startswith('plan_'))
    if 'observation' in data: observation_pose(data['observation'])
    if 'plan_report' in data:
        div=check_plan_report(data['plan_report'])
        a=data['plan_safety'];b=data['plan_corridor']
        if len(a)!=len(b) or any(abs(number(x['t_s'],'t')-number(y['t_s'],'t'))>1e-5 for x,y in zip(a,b)):
            raise ValueError('Planner sample times differ')
        d=max(math.hypot(pose(x,planning=True).x-pose(y,planning=True).x,
                         pose(x,planning=True).y-pose(y,planning=True).y) for x,y in zip(a,b))
        if abs(d-number(div['max_time_aligned_separation_m'],'divergence'))>.03:
            raise ValueError('Planner CSVs do not reproduce reported maximum separation')
        st=observation_pose(data['observation'])
        for k in ('plan_safety','plan_corridor'):
            first=pose(data[k][0],planning=True)
            if math.hypot(first.x-st.x,first.y-st.y)>.05:
                raise ValueError(f'{k} start does not align with stable CARLA observation')
    all_ego=[]
    for key in data:
        if key in ROOT_RELATIVE_PATHS and key not in ('observation','plan_report'):
            all_ego.extend([pose(r,planning=key.startswith('plan_')) for r in data[key]])
    if 'observation' in data: all_ego.append(observation_pose(data['observation']))
    if all_ego:
        ex=all_ego[0]
        for p in all_ego:
            if math.hypot(p.x-ex.x,p.y-ex.y)>70:
                raise ValueError('Unexpected mixed-map or far-apart source trajectories')
    return {'root':root,'paths':paths,'data':data,'scenes':included,'missing':missing,
            'source_hashes':{k:sha256(paths[k]) for k in data}}

def scene_seconds(name, data):
    if name in ('intro','outro'): return 2.0
    if name=='observation': return 2.5
    if name=='planning': return 3.0
    if name=='feedback': return 2.5
    if name in data:
        return number(data[name][-1]['t_s'],'last time')
    return 4.2

def scene_summary(name, data):
    labels={
      'intro': ('From maps to motion', 'CARLA + CommonRoad | visual showcase', 'All driving footage below is reconstructed from recorded telemetry.'),
      'observation': ('One road. One observed starting state.', 'Vehicle and cone placed from a settled CARLA observation', 'Map coordinates: CARLA y = -CommonRoad y | parked reconstruction'),
      'planning': ('Different priorities, different plans', 'Green: clearance-first  |  Orange: centerline-first', 'OFFLINE PROJECTED TRAJECTORIES — no physical obstacle avoidance'),
      'straight': ('Making the car move', 'Manual first gear enabled repeatable acceleration', 'RECORDED CARLA TRAJECTORY RECONSTRUCTION — not a new drive'),
      'steer_pos': ('Steering in one direction', 'CARLA steer +0.08  →  CommonRoad heading about -1.73°', 'RECORDED CARLA TRAJECTORY RECONSTRUCTION'),
      'steer_neg': ('Then the opposite direction', 'CARLA steer -0.08  →  CommonRoad heading about +1.73°', 'RECORDED CARLA TRAJECTORY RECONSTRUCTION'),
      'feedback': ('Feedback versus fixed steering', 'Position RMS: 16.30 mm without correction → 10.07 mm with correction', 'ARCHIVED TRACKING RESULTS — modified empirical reference'),
      'route_off': ('Driving a CommonRoad-derived route', 'Reference is created from Town01 route geometry, not a CARLA recording', 'RECORDED CARLA TELEMETRY — steering feedback OFF'),
      'route_on': ('Correcting route tracking', 'Same route, steering feedback ON; RMS 5.96 cm → 3.30 cm', 'RECORDED CARLA TELEMETRY — steering feedback ON'),
      'speed_failed': ('When speed control went wrong', 'A tiny coast-phase brake command triggered a sudden speed drop', 'ARCHIVED ABORTED RUN — safety limit stopped the experiment'),
      'speed_fixed': ('Fixing the coast-phase response', 'Zero braking while coasting restored a complete run', 'RECORDED COMPLETED RUN — not a fully calibrated speed controller'),
      'outro': ('What comes next', 'Better braking model, longer routes, eventual Rulebook and Scenic integration', 'Original Rulebook execution and Scenic scenes are future work.'),
    }
    return labels[name]
