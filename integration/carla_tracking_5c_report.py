"""Milestone 5C: compare historical 5A tracking with 5C position feedback.

Offline only: reads JSON + CSV; never imports CARLA or changes a simulator.
No claims of repeatability are made from a single run per condition.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

HERE = Path(__file__).resolve().parent
PROFILES = ('corridor_first', 'safety_first')
METRICS = ('position_rmse_m', 'max_position_error_m',
           'final_position_error_m', 'along_track_rmse_m',
           'max_abs_cross_track_error_m', 'max_abs_heading_error_deg',
           'speed_tracking_rmse_mps')


def read_summary(path: Path):
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('status') != 'completed':
        raise ValueError(f'{path}: expected completed run; got {data.get("status")!r}')
    if data.get('collision_event_count', 0):
        raise ValueError(f'{path}: collision events present; inspect run separately')
    required = ('source_sha256', 'fixed_dt_s', 'vehicle_blueprint',
                'world_map', 'samples', 'duration_s', 'lookahead_m')
    if any(k not in data for k in required):
        raise ValueError(f'{path}: missing required metadata')
    for k in ('position_rmse_m','max_abs_cross_track_error_m',
              'max_abs_heading_error_deg','max_position_error_m','final_position_error_m'):
        value=data.get(k)
        if not isinstance(value, (int,float)) or not math.isfinite(value):
            raise ValueError(f'{path}: missing or nonfinite {k}')
    # 5A summaries predate derived along-track / speed RMSE fields. Recover
    # these values from the 5A tracking CSV instead of inventing data.
    if data.get('along_track_rmse_m') is None or data.get('speed_tracking_rmse_mps') is None:
        csv_path=path.parent/'tracking_samples.csv'
        with csv_path.open('r',newline='',encoding='utf-8-sig') as f:
            rows=list(csv.DictReader(f))
        if len(rows) != int(data['samples']):
            raise ValueError(f'{path}: tracking CSV count does not match summary')
        try:
            along=[float(r['along_track_error_m']) for r in rows]
            speed=[float(r['ref_speed_mps'])-float(r['actual_speed_mps']) for r in rows]
        except (KeyError,ValueError,TypeError) as exc:
            raise ValueError(f'{csv_path}: missing numeric tracking columns') from exc
        if not all(math.isfinite(v) for v in along+speed):
            raise ValueError(f'{csv_path}: nonfinite tracked error')
        data['along_track_rmse_m']=math.sqrt(mean(v*v for v in along))
        data['speed_tracking_rmse_mps']=math.sqrt(mean(v*v for v in speed))
    for k in METRICS:
        if data[k] is None or not math.isfinite(float(data[k])):
            raise ValueError(f'{path}: invalid {k}')
    return data


def find_runs(root: Path, profile: str):
    folder=root/profile
    paths=[]
    if (folder/'summary.json').is_file():
        paths.append(folder/'summary.json')
    if folder.is_dir():
        paths.extend(sorted(folder.glob('run_*/summary.json')))
    if not paths:
        raise FileNotFoundError(f'No run summaries under {folder} (or run_*/summary.json)')
    return paths


def check_comparable(baseline, enhanced, basepath, enhancedpath):
    for k in ('source_sha256','vehicle_blueprint','world_map','samples',
              'fixed_dt_s','duration_s','lookahead_m','initial_velocity_set_once'):
        if baseline.get(k)!=enhanced.get(k):
            raise ValueError(f'Runs not comparable: field {k!r} differs between {basepath} and {enhancedpath}')
    if enhanced.get('controller_type') != 'position':
        raise ValueError(f'{enhancedpath}: expected position controller, got {enhanced.get("controller_type")!r}')


def aggregate(baseline_root: Path, enhanced_root: Path):
    rows=[]
    output={}
    for profile in PROFILES:
        original_paths=find_runs(baseline_root,profile)
        updated_paths=find_runs(enhanced_root,profile)
        original=[read_summary(p) for p in original_paths]
        updated=[read_summary(p) for p in updated_paths]
        for b,pb in zip(original,original_paths):
            for e,pe in zip(updated,updated_paths):
                check_comparable(b,e,pb,pe)
        output[profile]={'baseline_runs':len(original), 'enhanced_runs':len(updated),
                         'baseline_paths':[str(p) for p in original_paths],
                         'enhanced_paths':[str(p) for p in updated_paths],
                         'baseline_repeatability_estimated':len(original)>=2,
                         'enhanced_repeatability_estimated':len(updated)>=2,
                         'metrics':{}}
        for metric in METRICS:
            bvals=[float(x[metric]) for x in original]
            evals=[float(x[metric]) for x in updated]
            bm=mean(bvals);em=mean(evals)
            diff=em-bm
            gain=(100.0*(bm-em)/bm) if abs(bm)>1e-10 else None
            output[profile]['metrics'][metric]={
                'baseline_mean':bm,'enhanced_mean':em,
                'delta_enhanced_minus_baseline':diff,
                'improvement_percent_positive_is_better':gain,
                'baseline_sample_std':stdev(bvals) if len(bvals)>1 else None,
                'enhanced_sample_std':stdev(evals) if len(evals)>1 else None}
            rows.append({'profile':profile,'metric':metric,
                         'baseline_n':len(bvals),'enhanced_n':len(evals),
                         **output[profile]['metrics'][metric]})
    return rows,output


def plot_rmse(out: Path, overview):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    labels=['R5 first','R9 first']
    values=[overview[p]['metrics']['position_rmse_m'] for p in PROFILES]
    fig,ax=plt.subplots(figsize=(8,4.5))
    x=[0,1];width=.35
    ax.bar([v-width/2 for v in x],[r['baseline_mean'] for r in values],width,label='Milestone 5A baseline')
    ax.bar([v+width/2 for v in x],[r['enhanced_mean'] for r in values],width,label='Milestone 5C position feedback')
    ax.set_xticks(x,labels);ax.set_ylabel('Position RMSE (m); lower is better')
    ax.set_title('CARLA tracking: original vs position-aware feedback')
    ax.legend();ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/'tracking_rmse_comparison.png',dpi=150);plt.close(fig)
    return True


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root',type=Path,default=HERE/'output/carla_tracking')
    p.add_argument('--enhanced-root',type=Path,default=HERE/'output/carla_tracking_5c')
    p.add_argument('--out',type=Path,default=HERE/'output/carla_tracking_5c/report')
    args=p.parse_args(argv)
    rows,summary=aggregate(args.baseline_root,args.enhanced_root)
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'tracking_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    payload={'profiles':summary,'interpretation':
             'Offline comparison of separately executed precomputed plans. Fewer than two '
             'runs in any condition cannot estimate its repeatability. No obstacle in CARLA.'}
    payload['plot_written']=plot_rmse(args.out,summary)
    (args.out/'tracking_comparison.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print('===== MILESTONE 5C: CONTROLLER COMPARISON =====')
    for profile in PROFILES:
        q=summary[profile];r=q['metrics']['position_rmse_m']
        print(f'{profile}: baseline n={q["baseline_runs"]}, position n={q["enhanced_runs"]}; '
              f'RMSE {r["baseline_mean"]:.3f} -> {r["enhanced_mean"]:.3f} m')
        if q['baseline_runs']<2 or q['enhanced_runs']<2:
            print('  Repeatability NOT established: need >=2 independent runs per condition.')
    print('Results:',args.out)


if __name__=='__main__':
    main()
