"""Milestone 5B: compare paired CARLA executions with their offline references.

Offline analysis of Milestone 5A tracking_samples.csv files. This does NOT
simulate either vehicle, certify Rulebook compliance, or prove collision safety.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path

FIELDS = ('t_s', 'ref_x_cr_m', 'ref_y_cr_m', 'actual_x_cr_m',
          'actual_y_cr_m', 'ref_yaw_cr_deg')


def load_rows(path: Path):
    rows = []
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or any(field not in reader.fieldnames for field in FIELDS):
            raise ValueError(f'{path}: expected fields: {", ".join(FIELDS)}')
        for lineno, row in enumerate(reader, 2):
            try:
                record = {field: float(row[field]) for field in FIELDS}
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{path}:{lineno}: invalid numeric value') from exc
            if not all(map(math.isfinite, record.values())):
                raise ValueError(f'{path}:{lineno}: nonfinite value')
            rows.append(record)
    if len(rows) < 2:
        raise ValueError(f'{path}: need at least 2 samples')
    if any(b['t_s'] <= a['t_s'] for a, b in zip(rows, rows[1:])):
        raise ValueError(f'{path}: time must be strictly increasing')
    return rows


def interp(rows, times, t):
    if t < times[0] - 1e-7 or t > times[-1] + 1e-7:
        raise ValueError('Cannot extrapolate outside a logged trajectory')
    i = bisect.bisect_left(times, t)
    if i == 0:
        return rows[0]
    if i == len(times):
        return rows[-1]
    if abs(times[i] - t) < 1e-7:
        return rows[i]
    a, b = rows[i-1], rows[i]
    alpha = (t-a['t_s'])/(b['t_s']-a['t_s'])
    result = {'t_s': t}
    for k in FIELDS[1:]:
        if k == 'ref_yaw_cr_deg':
            # Short-way circular interpolation (handles +/- 180 deg crossings).
            delta = (b[k]-a[k] + 180.0) % 360.0 - 180.0
            result[k] = a[k] + alpha * delta
        else:
            result[k] = a[k] + alpha*(b[k]-a[k])
    return result


def calculate(r9, r5):
    times5 = [r['t_s'] for r in r5]
    lo = max(r9[0]['t_s'], times5[0])
    hi = min(r9[-1]['t_s'], times5[-1])
    if hi <= lo:
        raise ValueError('Runs have no overlapping time interval')
    grid = [r['t_s'] for r in r9 if lo-1e-7 <= r['t_s'] <= hi+1e-7]
    if len(grid) < 2:
        raise ValueError('Too few overlapping time samples')
    out = []
    for t in grid:
        a = interp(r9, [r['t_s'] for r in r9], t)
        b = interp(r5, times5, t)
        ppx = a['ref_x_cr_m'] - b['ref_x_cr_m']
        ppy = a['ref_y_cr_m'] - b['ref_y_cr_m']
        aax = a['actual_x_cr_m'] - b['actual_x_cr_m']
        aay = a['actual_y_cr_m'] - b['actual_y_cr_m']
        planned_d = math.hypot(ppx, ppy)
        actual_d = math.hypot(aax, aay)
        theta = math.radians(b['ref_yaw_cr_deg'])
        c, s = math.cos(theta), math.sin(theta)
        epx, epy = aax-ppx, aay-ppy
        out.append({
            't_s': round(t, 7),
            'planned_separation_m': planned_d,
            'executed_separation_m': actual_d,
            'separation_difference_m': actual_d-planned_d,
            'planned_along_difference_m': c*ppx+s*ppy,
            'executed_along_difference_m': c*aax+s*aay,
            'planned_lateral_difference_m': -s*ppx+c*ppy,
            'executed_lateral_difference_m': -s*aax+c*aay,
            'divergence_vector_error_m': math.hypot(epx,epy),
            'r9_tracking_error_m': math.hypot(a['actual_x_cr_m']-a['ref_x_cr_m'], a['actual_y_cr_m']-a['ref_y_cr_m']),
            'r5_tracking_error_m': math.hypot(b['actual_x_cr_m']-b['ref_x_cr_m'], b['actual_y_cr_m']-b['ref_y_cr_m'])
        })
    return out


def stats(rows):
    def mx(k): return max(r[k] for r in rows)
    def maxabs(k): return max(abs(r[k]) for r in rows)
    def rmse(k): return math.sqrt(sum(r[k]**2 for r in rows)/len(rows))
    return {
        'aligned_samples':len(rows),
        'time_window_s':[rows[0]['t_s'], rows[-1]['t_s']],
        'max_planned_separation_m':mx('planned_separation_m'),
        'max_executed_separation_m':mx('executed_separation_m'),
        'final_planned_separation_m':rows[-1]['planned_separation_m'],
        'final_executed_separation_m':rows[-1]['executed_separation_m'],
        'max_abs_planned_lateral_difference_m':maxabs('planned_lateral_difference_m'),
        'max_abs_executed_lateral_difference_m':maxabs('executed_lateral_difference_m'),
        'max_abs_planned_along_difference_m':maxabs('planned_along_difference_m'),
        'max_abs_executed_along_difference_m':maxabs('executed_along_difference_m'),
        'max_abs_scalar_separation_difference_m':maxabs('separation_difference_m'),
        'divergence_vector_error_rmse_m':rmse('divergence_vector_error_m'),
        'max_divergence_vector_error_m':mx('divergence_vector_error_m'),
        'r9_tracking_rmse_m':rmse('r9_tracking_error_m'),
        'r5_tracking_rmse_m':rmse('r5_tracking_error_m'),
        'interpretation':'Paired offline playback from separate CARLA runs; does not demonstrate live replanning or obstacle avoidance.'
    }


def plot(out, rows):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    t=[r['t_s'] for r in rows]
    fig,ax=plt.subplots(figsize=(8.6,4.7))
    ax.plot(t,[r['planned_separation_m'] for r in rows],label='Planned separation')
    ax.plot(t,[r['executed_separation_m'] for r in rows],label='Executed separation')
    ax.set(xlabel='Time (s)',ylabel='Body-center separation (m)',title='Milestone 5B: planned vs executed divergence')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'separation_over_time.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.6,4.7))
    ax.plot(t,[r['planned_lateral_difference_m'] for r in rows],label='Planned lateral difference')
    ax.plot(t,[r['executed_lateral_difference_m'] for r in rows],label='Executed lateral difference')
    ax.set(xlabel='Time (s)',ylabel='Signed lateral difference (m)',title='Milestone 5B: lateral behavior under CARLA execution')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'lateral_difference_over_time.png',dpi=160);plt.close(fig)
    return True


def main():
    here=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--r9',type=Path,default=here/'output/carla_tracking/safety_first/tracking_samples.csv')
    parser.add_argument('--r5',type=Path,default=here/'output/carla_tracking/corridor_first/tracking_samples.csv')
    parser.add_argument('--out',type=Path,default=here/'output/carla_tracking/paired_comparison')
    args=parser.parse_args()
    result=calculate(load_rows(args.r9),load_rows(args.r5))
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'divergence_samples.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(result[0]));writer.writeheader();writer.writerows(result)
    summary=stats(result)
    summary['source_r9_csv']=str(args.r9)
    summary['source_r5_csv']=str(args.r5)
    summary['plots_written']=plot(args.out,result)
    (args.out/'paired_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print('===== MILESTONE 5B: PAIRED CARLA DIVERGENCE =====')
    for key in ('aligned_samples','max_planned_separation_m','max_executed_separation_m',
                'final_planned_separation_m','final_executed_separation_m',
                'max_abs_planned_lateral_difference_m','max_abs_executed_lateral_difference_m',
                'divergence_vector_error_rmse_m','r9_tracking_rmse_m','r5_tracking_rmse_m'):
        print(f'{key}: {summary[key]}')
    print('Outputs:',args.out)
    print('LIMIT: separate tracking runs of fixed offline plans, synthetic obstacle absent.')


if __name__=='__main__':
    main()
