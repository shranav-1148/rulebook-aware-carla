"""Milestone 4C: controlled obstacle sweep for the M4B motion-graph prototype.

AI-authored experiment adapter. It imports the *unchanged* M4B
`town01_motion_graph.py`, creates ONE feasible graph for each obstacle
configuration, and performs two lexicographic searches on that SAME graph.

These are NOT the actual CommonRoad Rulebook's R5/R9 rule functions, nor a
validity-certified vehicle model, nor a CARLA/Scenic execution.
"""

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

from town01_motion_graph import (
    Settings, DEFAULT_ROUTE, build_lattice, divergence, maybe_plot,
    path_rows, read_route_csv, save_csv, select_trajectory, synthetic_route,
    unpack_state, world_xy,
)

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "output" / "town01_conflict_sweep"
DEFAULT_STATIONS = "12,21,30,42,65"
DEFAULT_OFFSETS = "-3.2,-2.4,-1.2,0,1.2,2.4,3.2"


def parse_numeric_values(spec, name):
    """Parse ordered, unique, finite comma-separated floats."""
    try:
        values = [float(x.strip()) for x in spec.split(",")]
    except ValueError as exc:
        raise ValueError(f"{name} must contain comma-separated numbers") from exc
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError(f"{name} must contain finite numbers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate values")
    return values


def classify_failure(exc):
    """Only expected graph-coverage failures are treated as infeasible cases."""
    msg = str(exc)
    if msg.startswith("No feasible states at step "):
        return "no_full_horizon_path"
    if msg == "No terminal state reachable":
        return "no_eligible_terminal"
    return None


def evaluate_case(route, cfg, min_final_s=None):
    """Evaluate one scene using identical feasible graph and terminal filter.

    A missing full-horizon path is recorded explicitly; we never count it as
    zero divergence or as an argument for either priority structure.
    """
    result = {
        "obstacle_station_m": cfg.obstacle_station,
        "obstacle_offset_m": cfg.obstacle_offset,
        "status": "ok",
        "reason": "",
        "states": None,
        "edges": None,
        "eligible_terminals": None,
        "obstacle_center_xy_m": list(world_xy(route, cfg.obstacle_station, cfg.obstacle_offset)),
        "same_feasible_graph": True,
        "same_terminal_set": True,
        "same_selected_state_path": None,
        "R9_first_R9": None,
        "R9_first_R5": None,
        "R9_first_R4": None,
        "R5_first_R9": None,
        "R5_first_R5": None,
        "R5_first_R4": None,
        "R9_first_final_s_m": None,
        "R5_first_final_s_m": None,
        "max_aligned_separation_m": None,
        "mean_aligned_separation_m": None,
        "final_separation_m": None,
        "max_lateral_offset_difference_m": None,
        "max_station_difference_m": None,
        "final_station_difference_m": None,
    }
    try:
        start, graph, terminals, layers, obstacle = build_lattice(route, cfg)
    except RuntimeError as exc:
        status = classify_failure(exc)
        if status is None:
            raise
        result.update(status=status, reason=str(exc))
        return result, None
    result["states"] = sum(layer["states"] for layer in layers) + 1
    result["edges"] = sum(layer["edges_from_previous"] for layer in layers)
    result["obstacle_center_xy_m"] = list(obstacle)

    if min_final_s is not None:
        terminals = [s for s in terminals if unpack_state(s, cfg)[1] >= min_final_s - 1e-9]
    result["eligible_terminals"] = len(terminals)
    if not terminals:
        result.update(status="no_eligible_terminal", reason=f"No final state reaches s >= {min_final_s} m")
        return result, None

    # Crucially, this is one exact graph and one terminal list, reused for both.
    path_a, actions_a, cost_a = select_trajectory(start, graph, terminals, "safety_first", cfg)
    path_b, actions_b, cost_b = select_trajectory(start, graph, terminals, "corridor_first", cfg)
    rows_a = path_rows(path_a, actions_a, cost_a, "safety_first", cfg, route)
    rows_b = path_rows(path_b, actions_b, cost_b, "corridor_first", cfg, route)
    delta = divergence(rows_a, rows_b)

    # Both priority profiles see identical candidates. Each must minimize its
    # respective first cost, subject only to float tolerance and tie-rounding.
    tol = 1e-7
    if cost_a[0] > cost_b[0] + tol or cost_b[1] > cost_a[1] + tol:
        raise AssertionError("Primary-priority optimality check failed")

    result.update({
        "same_selected_state_path": path_a == path_b,
        "R9_first_R9": cost_a[0], "R9_first_R5": cost_a[1], "R9_first_R4": cost_a[2],
        "R5_first_R9": cost_b[0], "R5_first_R5": cost_b[1], "R5_first_R4": cost_b[2],
        "R9_first_final_s_m": rows_a[-1]["s_m"],
        "R5_first_final_s_m": rows_b[-1]["s_m"],
        **delta,
    })
    return result, {"R9_first": rows_a, "R5_first": rows_b, "obstacle": obstacle, "cfg": cfg}


def run_sweep(route, base_cfg, stations, offsets, min_final_s=None):
    """Return records plus detailed trajectories for selected diagnostics."""
    all_records, successful_details = [], {}
    for station in stations:
        for offset in offsets:
            cfg = replace(base_cfg, obstacle_station=station, obstacle_offset=offset)
            record, detail = evaluate_case(route, cfg, min_final_s)
            all_records.append(record)
            if detail is not None:
                successful_details[(station, offset)] = detail
    return all_records, successful_details


def save_records(path, records):
    if not records:
        raise ValueError("Cannot save an empty sweep")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        for row in records:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})


def plot_heatmaps(out, records, stations, offsets):
    """Optional visualizations. Infeasible scenes are shown as missing, not zero."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return []
    output = []
    datasets = [
        ("max_aligned_separation_m", "max_separation_heatmap.png", "Maximum time-aligned separation (m)"),
        ("max_lateral_offset_difference_m", "lateral_difference_heatmap.png", "Maximum lateral-offset difference (m)"),
        ("final_station_difference_m", "progress_difference_heatmap.png", "Final absolute station difference (m)"),
    ]
    lookup = {(r["obstacle_station_m"], r["obstacle_offset_m"]): r for r in records}
    for key, filename, title in datasets:
        vals = [[(lookup[(s, d)][key] if lookup[(s, d)]["status"] == "ok" else float("nan"))
                 for s in stations] for d in offsets]
        arr = np.ma.masked_invalid(np.asarray(vals, dtype=float))
        fig, ax = plt.subplots(figsize=(max(7, 1.15 * len(stations)), max(4.8, 0.65 * len(offsets))))
        im = ax.imshow(arr, origin="lower", aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(stations)), [f"{s:g}" for s in stations])
        ax.set_yticks(range(len(offsets)), [f"{d:g}" for d in offsets])
        ax.set_xlabel("Obstacle station along route (m)")
        ax.set_ylabel("Obstacle lateral offset from centerline (m; left positive)")
        ax.set_title(title + " — missing cells = infeasible")
        for yi, d in enumerate(offsets):
            for xi, s in enumerate(stations):
                r = lookup[(s, d)]
                cell_label = f"{r[key]:.1f}" if r["status"] == "ok" else "X"
                ax.text(xi, yi, cell_label, ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, label=title)
        fig.tight_layout()
        dest = out / filename
        fig.savefig(dest, dpi=160)
        plt.close(fig)
        output.append(dest.name)
    return output


def save_examples(out, route, records, details):
    """Export at most three representative paired trajectories."""
    if not details:
        return []
    ok = [r for r in records if r["status"] == "ok"]
    max_div = max(ok, key=lambda r: r["max_aligned_separation_m"])
    examples = []
    baseline = next((r for r in ok if r["obstacle_station_m"] == 21.0
                     and r["obstacle_offset_m"] == -2.4), None)
    if baseline is not None:
        examples.append(("baseline_21m_right_2p4m", baseline))
    examples.append(("largest_divergence", max_div))
    same = next((r for r in ok if r["same_selected_state_path"]), None)
    if same is not None:
        examples.append(("identical_decisions", same))

    created = []
    used_cases = set()
    for name, r in examples:
        key = (r["obstacle_station_m"], r["obstacle_offset_m"])
        if key in used_cases:
            continue
        used_cases.add(key)
        item = details[key]
        folder = out / "examples" / name
        folder.mkdir(parents=True, exist_ok=True)
        save_csv(folder / "R9_first.csv", item["R9_first"])
        save_csv(folder / "R5_first.csv", item["R5_first"])
        maybe_plot(folder / "comparison.png", route, item["obstacle"],
                   item["R9_first"], item["R5_first"], item["cfg"])
        (folder / "case_summary.json").write_text(json.dumps(r, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        created.append(str(folder.relative_to(out)))
    return created


def execute(route, route_source, cfg, stations, offsets, out, min_final_s=None, route_hash=None):
    out.mkdir(parents=True, exist_ok=True)
    records, details = run_sweep(route, cfg, stations, offsets, min_final_s)
    save_records(out / "scenario_results.csv", records)
    plots = plot_heatmaps(out, records, stations, offsets)
    examples = save_examples(out, route, records, details)
    ok = [r for r in records if r["status"] == "ok"]
    diff = [r for r in ok if not r["same_selected_state_path"]]
    summary = {
        "experiment": "Milestone 4C: controlled obstacle-position sweep of AI-authored M4B prototype",
        "route_source": route_source,
        "route_sha256": route_hash,
        "settings": asdict(cfg),
        "stations_m": stations,
        "offsets_m_left_positive": offsets,
        "minimum_terminal_station_m": min_final_s,
        "cases_total": len(records),
        "cases_full_horizon_feasible": len(ok),
        "cases_infeasible_or_no_terminal": len(records) - len(ok),
        "cases_with_different_selected_state_path": len(diff),
        "largest_time_aligned_separation_m": max((r["max_aligned_separation_m"] for r in ok), default=None),
        "status_counts": {status: sum(r["status"] == status for r in records)
                          for status in sorted({r["status"] for r in records})},
        "outputs": ["scenario_results.csv", "sweep_summary.json"] + plots + examples,
        "caveats": [
            "This evaluates Rulebook-inspired lexicographic TOTAL priorities, not the original partially ordered Rulebook algorithm.",
            "R9/R5/R4 are illustrative proxy costs, not upstream rule formulas.",
            "Obstacle positions are synthetic and constant; the Town01 route supplies geometry only.",
            "A single common graph is used for both priorities WITHIN each obstacle case; graphs may differ ACROSS cases.",
            "No verified lane boundaries, yaw/steering/bicycle dynamics, or continuous collision checking.",
            "Failures are excluded from divergence statistics, not assigned zero divergence.",
            "A terminal progress gate is not proof of reaching a common destination or matching longitudinal position.",
            "No CARLA or Scenic execution has been performed.",
        ],
    }
    (out / "sweep_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("\n===== MILESTONE 4C: CONTROLLED PRIORITY-CONFLICT SWEEP =====")
    print("Route:", route_source)
    print("Obstacle stations (m):", stations)
    print("Obstacle lateral offsets (m, left positive):", offsets)
    print("Minimum final progress (m):", min_final_s if min_final_s is not None else "none")
    print(f"Cases: {len(records)} total, {len(ok)} full-horizon feasible, {len(records)-len(ok)} infeasible/no terminal")
    print(f"Selected paths differ: {len(diff)} / {len(ok)} feasible cases")
    if ok:
        max_case = max(ok, key=lambda r: r["max_aligned_separation_m"])
        print("Max time-aligned separation: {:.3f} m (obstacle s={:g} m, d={:g} m)".format(
            max_case["max_aligned_separation_m"], max_case["obstacle_station_m"], max_case["obstacle_offset_m"]))
    print("Status counts:", summary["status_counts"])
    print("CSV:", out / "scenario_results.csv")
    print("Summary:", out / "sweep_summary.json")
    if plots:
        print("Heatmaps:", ", ".join(plots))
    else:
        print("Heatmaps skipped (matplotlib/numpy unavailable)")
    print("Example trajectory folders:", ", ".join(examples) or "none")
    print("NOTE: Proxy costs and discrete graph only; not actual Rulebook or CARLA execution.")
    return summary, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", type=Path, default=DEFAULT_ROUTE, help="Milestone 4A route_waypoints.csv")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--demo", action="store_true", help="Use synthetic straight route instead of Town01 CSV")
    parser.add_argument("--stations", default=DEFAULT_STATIONS, help="Comma-separated obstacle stations in meters")
    parser.add_argument("--offsets", default=DEFAULT_OFFSETS, help="Comma-separated lateral offsets in meters (left positive)")
    parser.add_argument("--soft-clearance", type=float, default=Settings().soft_clearance)
    parser.add_argument("--min-final-s", type=float, default=None,
                        help="Require both searches to choose from identical terminal states with s >= this value")
    args = parser.parse_args()
    stations = parse_numeric_values(args.stations, "stations")
    offsets = parse_numeric_values(args.offsets, "offsets")
    if args.soft_clearance < 0 or not math.isfinite(args.soft_clearance):
        parser.error("--soft-clearance must be finite and >= 0")
    if args.min_final_s is not None and (args.min_final_s < 0 or not math.isfinite(args.min_final_s)):
        parser.error("--min-final-s must be finite and >= 0")
    route = synthetic_route() if args.demo else read_route_csv(args.route)
    if any(s <= 0 or s + Settings().initial_speed >= route[-1][0] for s in stations):
        parser.error("All obstacle stations must lie within the route, leaving some room afterward")
    cfg = replace(Settings(), soft_clearance=args.soft_clearance)
    source = "synthetic-demo" if args.demo else str(args.route)
    digest = None if args.demo else hashlib.sha256(args.route.read_bytes()).hexdigest()
    execute(route, source, cfg, stations, offsets, args.out, args.min_final_s, digest)


if __name__ == "__main__":
    main()
