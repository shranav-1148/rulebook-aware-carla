"""Compare a Milestone 5C R9-first run with the 5C-B coast-first trial.

Offline only: reads two real CARLA execution logs; does not import CARLA.
Compares the early speed/brake transient and global tracking statistics.
Does not infer cause from observed correlation and does not establish repeatability.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE/'output/carla_tracking_5c/safety_first'
TRIAL=HERE/'output/carla_tracking_5cb/safety_first'
OUT=HERE/'output/carla_tracking_5cb/brake_transition_report'


def read_run(folder:Path):
    summary_path=folder/'summary.json'
    csv_path=folder/'tracking_samples.csv'
    summary=json.loads(summary_path.read_text(encoding='utf-8'))
    if summary.get('status')!='completed' or summary.get('collision_event_count')!=0:
        raise ValueError(f'Run not completed/collision-free: {folder}')
    with csv_path.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        needed=('t_s','ref_speed_mps','actual_speed_mps','control_brake','control_throttle','position_error_m')
        if not set(needed).issubset(reader.fieldnames or []):
            raise ValueError(f'CSV lacks required columns in {folder}')
        rows=list(reader)
    if len(rows)<2 or len(rows)!=summary.get('samples'):
        raise ValueError(f'CSV row count does not match summary in {folder}')
    for row in rows:
        for key in needed:
            v=float(row[key])
            if not math.isfinite(v):
                raise ValueError(f'Nonfinite {key} in {folder}')
    times=[float(r['t_s']) for r in rows]
    if abs(times[0])>1e-7 or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError(f'Invalid time axis in {folder}')
    return summary,rows


def verify_compatible(base,trial):
    if base.get('controller_type')!='position' or trial.get('controller_type')!='coast':
        raise ValueError('Expected original 5C position and new 5C-B coast controllers')
    fields=('source_sha256','vehicle_blueprint','carla_client_version','carla_server_version',
            'fixed_dt_s','lookahead_m','wheelbase_assumed_m','initial_velocity_set_once',
            'world_map','synthetic_obstacle_spawned')
    for key in fields:
        if base.get(key)!=trial.get(key):
            raise ValueError(f'Mismatch in {key}: {base.get(key)!r} vs {trial.get(key)!r}')
    for key in ('position_kp_per_second','max_speed_correction_mps','max_commanded_speed_mps'):
        if not math.isclose(float(base.get(key)),float(trial.get(key)),rel_tol=1e-9,abs_tol=1e-9):
            raise ValueError(f'Controller position feedback setting mismatch: {key}')
    if base.get('synthetic_obstacle_spawned') is not False:
        raise ValueError('Expected obstacle-free tracking comparison')


def metrics(rows,early_end=1.5):
    pairs=[(float(r['t_s']),float(r['actual_speed_mps']),float(r['control_brake'])) for r in rows]
    early=[x for x in pairs if 0 < x[0]<=early_end]
    if not early:
        raise ValueError('No samples within early transient window')
    delta=[(t, (v-vprev)/(t-tprev)) for (tprev,vprev,_),(t,v,_) in zip(pairs,pairs[1:])]
    sharp=[{'t_s':round(t,4),'apparent_speed_change_rate_mps2':round(rate,4)}
           for t,rate in delta if rate < -15]
    strongest=min(delta,key=lambda p:p[1])
    return {'early_window_s':early_end,
            'min_early_speed_mps':min(v for _,v,_ in early),
            'peak_early_brake':max(b for _,_,b in early),
            'early_brake_ticks':sum(b>1e-8 for _,_,b in early),
            'max_per_tick_speed_drop_mps':max(0.,max(vprev-v for (_,vprev,_),(_,v,_) in zip(pairs,pairs[1:]))),
            'most_negative_apparent_speed_change_rate_mps2':strongest[1],
            'time_of_largest_speed_drop_s':strongest[0],
            'severe_speed_drop_events':sharp}


def optional_plot(out:Path,br,cr):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig,axs=plt.subplots(2,1,figsize=(10,7),sharex=True)
    for rows,label in ((br,'Original 5C (position)'),(cr,'New 5C-B (coast)')):
        t=[float(r['t_s']) for r in rows]
        axs[0].plot(t,[float(r['actual_speed_mps']) for r in rows],label=label)
        axs[1].step(t,[float(r['control_brake']) for r in rows],where='post',label=label)
    t=[float(r['t_s']) for r in br]
    axs[0].plot(t,[float(r['ref_speed_mps']) for r in br],ls=':',label='Planned speed',color='gray')
    axs[0].set_ylabel('Speed (m/s)')
    axs[1].set_ylabel('Commanded brake [0..1]')
    axs[1].set_xlabel('Time since tracking start (s)')
    for ax in axs:
        ax.set_xlim(0,1.5)
        ax.grid(alpha=.25);ax.legend()
    fig.suptitle('Milestone 5C-B: first-1.5s brake-transition comparison')
    fig.tight_layout()
    fig.savefig(out/'brake_transition_compare.png',dpi=160)
    plt.close(fig)
    return True


def compare(basedir:Path,trialdir:Path,out:Path):
    base,br=read_run(basedir)
    trial,cr=read_run(trialdir)
    verify_compatible(base,trial)
    if len(br)!=len(cr) or any(abs(float(a['t_s'])-float(b['t_s']))>1e-6 for a,b in zip(br,cr)):
        raise ValueError('Time axes differ; runs must use matching fixed time steps and duration')
    out.mkdir(parents=True,exist_ok=True)
    report={'source_5c_position_run':str(basedir.resolve()),
            'source_5cb_coast_run':str(trialdir.resolve()),
            'sha256_reference':base['source_sha256'],
            'n_runs_per_condition':1,
            'repeatability_established':False,
            'original_5c':{**metrics(br),
                           'position_rmse_m':base['position_rmse_m'],
                           'final_position_error_m':base['final_position_error_m'],
                           'max_cross_track_error_m':base['max_abs_cross_track_error_m']},
            'coast_5cb':{**metrics(cr),
                         'position_rmse_m':trial['position_rmse_m'],
                         'final_position_error_m':trial['final_position_error_m'],
                         'max_cross_track_error_m':trial['max_abs_cross_track_error_m']},
            'interpretation':'Diagnostic comparison only; no causal attribution or repeatability from one run per controller.'}
    report['position_rmse_change_m']=trial['position_rmse_m']-base['position_rmse_m']
    report['graph_written']=optional_plot(out,br,cr)
    (out/'brake_transition_summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('===== MILESTONE 5C-B: BRAKE TRANSITION COMPARISON =====')
    for label,d in (('Original 5C',report['original_5c']),('Coast-first 5C-B',report['coast_5cb'])):
        print(f'{label}: early min speed={d["min_early_speed_mps"]:.3f} m/s; '
              f'early peak brake={d["peak_early_brake"]:.3f}; '
              f'largest per-tick speed drop={d["max_per_tick_speed_drop_mps"]:.3f} m/s; '
              f'position RMSE={d["position_rmse_m"]:.3f} m')
    print('RMSE change (coast minus original): %.3f m' % report['position_rmse_change_m'])
    print('NOTE: one trial per condition; unusually large speed drops still require investigation.')
    print('Outputs:',out)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=BASE)
    p.add_argument('--trial',type=Path,default=TRIAL)
    p.add_argument('--out',type=Path,default=OUT)
    args=p.parse_args(argv)
    try:
        compare(args.baseline,args.trial,args.out)
    except (ValueError,OSError,KeyError) as exc:
        p.error(str(exc))

if __name__=='__main__':
    main()
