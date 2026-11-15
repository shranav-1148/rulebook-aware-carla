"""Milestone 4B: bounded, discrete motion-lattice Rulebook *prototype*.

Requires only Python stdlib for graph generation and optimization. Uses the
Town01 route_waypoints.csv exported by our Milestone 4A script.

States include time, longitudinal station, speed, lateral offset, and lateral
speed. Edges integrate constant longitudinal/lateral accelerations for 1 s.
A circle proxy and synthetic lateral corridor bound are assumed. This is NOT
an Ackermann/bicycle-model-feasible or verified drivable trajectory.
"""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from bisect import bisect_right

HERE = Path(__file__).resolve().parent
DEFAULT_ROUTE = HERE / "output" / "town01_graph" / "route_waypoints.csv"
DEFAULT_OUT = HERE / "output" / "town01_motion_graph"


@dataclass(frozen=True)
class Settings:
    dt: float = 1.0
    horizon: int = 6
    initial_speed: float = 6.0
    speed_min: float = 5.0
    speed_max: float = 8.0
    ax_actions: tuple = (-1.0, 0.0, 1.0)
    ay_actions: tuple = (-0.5, 0.0, 0.5)
    max_abs_offset: float = 1.4
    max_abs_lateral_speed: float = 1.0
    ds_quantum: float = 0.5
    d_quantum: float = 0.25
    v_quantum: float = 1.0
    vy_quantum: float = 0.5
    obstacle_station: float = 21.0
    obstacle_offset: float = -2.4  # Right side of centerline
    ego_radius: float = 0.85
    obstacle_radius: float = 0.65
    soft_clearance: float = 1.6
    edge_samples: int = 5


# Key t, s_quantum, d_quantum, v_quantum, vy_quantum. Quantization is
# explicit: every next state is reprojected onto this fixed lattice.
# Reference positions and headings are recovered from the route centerline.


def quantize(x, resolution):
    return int(round(x / resolution))


def unpack_state(key, cfg):
    t, si, di, vi, vyi = key
    return (t, si * cfg.ds_quantum, di * cfg.d_quantum,
            vi * cfg.v_quantum, vyi * cfg.vy_quantum)


def initial_state(cfg):
    return (0, 0, 0, quantize(cfg.initial_speed, cfg.v_quantum), 0)


def read_route_csv(path):
    """Read M4A waypoint CSV, removing consecutive duplicate endpoints.

    Returns [(cumulative_s, x, y), ...] in travel order.
    """
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2:
        raise ValueError("At least 2 waypoints required")
    points = []
    for row in rows:
        xy = (float(row["x_m"]), float(row["y_m"]))
        if not all(math.isfinite(v) for v in xy):
            raise ValueError("Nonfinite waypoint")
        if not points or math.hypot(xy[0] - points[-1][1], xy[1] - points[-1][2]) > 1e-8:
            s = 0.0 if not points else points[-1][0] + math.hypot(
                xy[0] - points[-1][1], xy[1] - points[-1][2])
            points.append((s, xy[0], xy[1]))
    if len(points) < 2 or points[-1][0] <= 0:
        raise ValueError("Degenerate route")
    return points


def pose_at(route, s):
    """Arc-length interpolation; tangent gives heading, not an actual vehicle yaw."""
    if s < 0 or s > route[-1][0] + 1e-8:
        raise ValueError(f"Station {s:.3f} outside [0,{route[-1][0]:.3f}]")
    s = min(s, route[-1][0])
    ss = [p[0] for p in route]
    i = max(0, min(bisect_right(ss, s) - 1, len(route) - 2))
    a, b = route[i], route[i + 1]
    f = (s - a[0]) / (b[0] - a[0])
    x = a[1] + f * (b[1] - a[1])
    y = a[2] + f * (b[2] - a[2])
    yaw = math.atan2(b[2] - a[2], b[1] - a[1])
    return x, y, yaw


def world_xy(route, s, offset):
    x, y, yaw = pose_at(route, s)
    return (x - math.sin(yaw) * offset, y + math.cos(yaw) * offset)


def next_state(key, ax, ay, cfg, route_length):
    t, s, d, v, vy = unpack_state(key, cfg)
    dt = cfg.dt
    vn = v + ax * dt
    vyn = vy + ay * dt
    if not (cfg.speed_min - 1e-9 <= vn <= cfg.speed_max + 1e-9):
        return None
    if abs(vyn) > cfg.max_abs_lateral_speed + 1e-9:
        return None
    sn = s + v * dt + 0.5 * ax * dt * dt
    dn = d + vy * dt + 0.5 * ay * dt * dt
    # Check continuous lateral offset extrema (linear lateral velocity).
    if max(abs(d), abs(dn)) > cfg.max_abs_offset + 1e-9:
        return None
    if sn > route_length + 1e-9:
        return None
    nk = (t + 1, quantize(sn, cfg.ds_quantum),
          quantize(dn, cfg.d_quantum), quantize(vn, cfg.v_quantum),
          quantize(vyn, cfg.vy_quantum))
    _, sr, dr, vr, vyr = unpack_state(nk, cfg)
    # Avoid rounding into invalid or unphysical states.
    if sr > route_length + 1e-9 or abs(dr) > cfg.max_abs_offset + 1e-9:
        return None
    if abs(vr - vn) > 1e-8 or abs(vyr - vyn) > 1e-8:
        return None
    return nk


def edge_geometry_and_cost(key, nxt, ax, ay, cfg, route, obstacle_xy):
    """Check sampled swept-circle collision and calculate additive edge costs.

    Returns None if collision or assumed lateral-corridor bound is violated.
    R9 = integrated soft clearance shortfall, R5 = integrated d^2.
    Note sampling does NOT rigorously certify collision freedom.
    """
    _, s, d, v, vy = unpack_state(key, cfg)
    dt = cfg.dt
    clearance_goal = cfg.ego_radius + cfg.obstacle_radius + cfg.soft_clearance
    rsum = cfg.ego_radius + cfg.obstacle_radius
    n = cfg.edge_samples
    r9 = r5 = 0.0
    for j in range(n + 1):
        tau = dt * j / n
        st = s + v * tau + 0.5 * ax * tau ** 2
        lat = d + vy * tau + 0.5 * ay * tau ** 2
        if abs(lat) > cfg.max_abs_offset + 1e-9:
            return None
        xy = world_xy(route, st, lat)
        sep = math.hypot(xy[0] - obstacle_xy[0], xy[1] - obstacle_xy[1])
        if sep < rsum - 1e-8:
            return None
        weight = 0.5 if j in (0, n) else 1.0
        r9 += weight * max(0.0, clearance_goal - sep)
        r5 += weight * lat ** 2
    r9 *= dt / n
    r5 *= dt / n
    # Dimensionless illustrative comfort score.
    r4 = dt * (ax ** 2 + ay ** 2)
    return (r9, r5, r4)


def validate_settings(cfg, route):
    if cfg.dt <= 0 or cfg.horizon < 1 or cfg.edge_samples < 2:
        raise ValueError("Positive dt/horizon and at least 2 edge samples required")
    if cfg.ds_quantum <= 0 or cfg.d_quantum <= 0 or cfg.v_quantum <= 0 or cfg.vy_quantum <= 0:
        raise ValueError("Invalid lattice resolutions")
    if not cfg.speed_min <= cfg.initial_speed <= cfg.speed_max:
        raise ValueError("Initial speed outside bounds")
    if abs(cfg.initial_speed / cfg.v_quantum - round(cfg.initial_speed / cfg.v_quantum)) > 1e-8:
        raise ValueError("Initial speed must lie on the speed lattice")
    if route[-1][0] <= cfg.obstacle_station + cfg.initial_speed:
        raise ValueError("Route too short for synthetic obstacle")
    if not 0 < cfg.obstacle_station < route[-1][0]:
        raise ValueError("Obstacle station outside route")


def build_lattice(route, cfg):
    """Build ONE feasible directed acyclic state-transition graph for both Rulebooks."""
    validate_settings(cfg, route)
    start = initial_state(cfg)
    obstacle = world_xy(route, cfg.obstacle_station, cfg.obstacle_offset)
    reachable = {start}
    layer = {start}
    edges = {}  # state -> [(next_state, (R9,R5,R4), (ax,ay)), ...]
    counts = []
    for step in range(cfg.horizon):
        new_layer = set()
        n_edges = 0
        for state in sorted(layer):
            outs = []
            for ax in cfg.ax_actions:
                for ay in cfg.ay_actions:
                    nxt = next_state(state, ax, ay, cfg, route[-1][0])
                    if nxt is None:
                        continue
                    vals = edge_geometry_and_cost(state, nxt, ax, ay, cfg, route, obstacle)
                    if vals is None:
                        continue
                    outs.append((nxt, vals, (ax, ay)))
                    new_layer.add(nxt)
                    n_edges += 1
            edges[state] = outs
        reachable.update(new_layer)
        counts.append({"time_index": step + 1, "states": len(new_layer), "edges_from_previous": n_edges})
        layer = new_layer
        if not layer:
            raise RuntimeError(f"No feasible states at step {step+1}")
    return start, edges, sorted(layer), counts, obstacle


def cost_rank(cost, profile):
    # Total cost tuple: (R9, R5, R4); R7 progress is terminal tie-breaker.
    r9, r5, r4 = cost
    if profile == "safety_first":
        return (round(r9, 10), round(r5, 10), round(r4, 10))
    if profile == "corridor_first":
        return (round(r5, 10), round(r9, 10), round(r4, 10))
    raise ValueError("Unknown priority profile")


def select_trajectory(start, graph, terminal, profile, cfg):
    """DAG shortest-path DP with lexicographically ordered additive costs.

    One label per state is sufficient because edge costs are additive and
    future possibilities depend only on the state. Progress is a terminal
    tie-breaker after the two tradeoff rules, before comfort.
    """
    best = {start: (0.0, 0.0, 0.0)}
    parents = {}
    states_by_time = {}
    for key in graph:
        states_by_time.setdefault(key[0], []).append(key)
    for k in range(cfg.horizon):
        for key in sorted(states_by_time.get(k, [])):
            if key not in best:
                continue
            old = best[key]
            for nxt, step_cost, action in graph[key]:
                new = tuple(old[i] + step_cost[i] for i in range(3))
                if nxt not in best or cost_rank(new, profile) < cost_rank(best[nxt], profile):
                    best[nxt] = new
                    parents[nxt] = (key, action)
    eligible = [st for st in terminal if st in best]
    if not eligible:
        raise RuntimeError("No terminal state reachable")
    # R7 progress as tie-breaker, R4 comfort after progress. R7 is not a
    # priority in the tradeoff pair, and never overrides R9/R5.
    if profile == "safety_first":
        kf = lambda s: (round(best[s][0], 10), round(best[s][1], 10),
                        -unpack_state(s, cfg)[1], round(best[s][2], 10), s)
    else:
        kf = lambda s: (round(best[s][1], 10), round(best[s][0], 10),
                        -unpack_state(s, cfg)[1], round(best[s][2], 10), s)
    final = min(eligible, key=kf)
    path = [final]
    actions = []
    while path[-1] != start:
        parent, action = parents[path[-1]]
        actions.append(action)
        path.append(parent)
    return list(reversed(path)), list(reversed(actions)), best[final]


def path_rows(states, actions, cost, profile, cfg, route):
    rows = []
    for i, state in enumerate(states):
        t, s, d, v, vy = unpack_state(state, cfg)
        x, y = world_xy(route, s, d)
        rows.append({"profile": profile, "t_s": t * cfg.dt, "s_m": s,
                     "d_m": d, "v_mps": v, "lateral_v_mps": vy,
                     "x_m": x, "y_m": y,
                     "ax_mps2": actions[i][0] if i < len(actions) else "",
                     "ay_mps2": actions[i][1] if i < len(actions) else ""})
    return rows


def divergence(a, b):
    if len(a) != len(b):
        raise ValueError("Different time lengths")
    distances = [math.hypot(x["x_m"] - y["x_m"], x["y_m"] - y["y_m"])
                 for x, y in zip(a, b)]
    lateral_differences = [abs(x["d_m"] - y["d_m"]) for x, y in zip(a, b)]
    progress_differences = [abs(x["s_m"] - y["s_m"]) for x, y in zip(a, b)]
    return {"max_aligned_separation_m": max(distances),
            "final_separation_m": distances[-1],
            "mean_aligned_separation_m": sum(distances) / len(distances),
            "max_lateral_offset_difference_m": max(lateral_differences),
            "max_station_difference_m": max(progress_differences),
            "final_station_difference_m": abs(a[-1]["s_m"] - b[-1]["s_m"])}


def save_csv(filename, rows):
    if not rows:
        raise ValueError("No rows")
    with filename.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(filename, route, obstacle, safety, corridor, cfg):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(11, 4.5))
    visible_route = [p for p in route if p[0] <= cfg.obstacle_station + 35.0]
    ax.plot([p[1] for p in visible_route], [p[2] for p in visible_route], "--", color="0.5", label="Lanelet-centerline route")
    ax.plot([p["x_m"] for p in safety], [p["y_m"] for p in safety], "o-", label="R9 first")
    ax.plot([p["x_m"] for p in corridor], [p["y_m"] for p in corridor], "s-", label="R5 first")
    ax.add_patch(Circle(obstacle, cfg.obstacle_radius, fc="tab:orange", alpha=0.5, label="Synthetic obstacle circle"))
    ax.add_patch(Circle(obstacle, cfg.ego_radius + cfg.obstacle_radius + cfg.soft_clearance, fill=False, ls=":", ec="tab:red", lw=1.2, label="Soft clearance boundary (ego center)"))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Motion-lattice demonstration: same graph, different priority order")
    ax.set_xlabel("CommonRoad X (m)"); ax.set_ylabel("CommonRoad Y (m)")
    ax.grid(alpha=0.2); ax.legend(loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(filename, dpi=155); plt.close(fig)
    return True


def synthetic_route(length=80.0, step=5.0):
    """Straight westbound demo route in CommonRoad coordinates (not Town01 data)."""
    n = int(length / step)
    return [(i * step, length - i * step, 0.0) for i in range(n + 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", type=Path, default=DEFAULT_ROUTE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--demo", action="store_true", help="Use synthetic straight polyline, not actual Town01")
    args = parser.parse_args()
    route = synthetic_route() if args.demo else read_route_csv(args.route)
    cfg = Settings()
    start, graph, terminals, layers, obstacle = build_lattice(route, cfg)
    sa, aa, ca = select_trajectory(start, graph, terminals, "safety_first", cfg)
    sb, ab, cb = select_trajectory(start, graph, terminals, "corridor_first", cfg)
    rows_a = path_rows(sa, aa, ca, "safety_first", cfg, route)
    rows_b = path_rows(sb, ab, cb, "corridor_first", cfg, route)
    delta = divergence(rows_a, rows_b)
    args.out.mkdir(parents=True, exist_ok=True)
    save_csv(args.out / "safety_first_trajectory.csv", rows_a)
    save_csv(args.out / "corridor_first_trajectory.csv", rows_b)
    maybe_plot(args.out / "comparison.png", route, obstacle, rows_a, rows_b, cfg)
    summary = {
        "experiment": "educational motion-lattice / lexicographic Rulebook-inspired prototype",
        "route_source": "synthetic-demo" if args.demo else str(args.route),
        "route_sha256": None if args.demo else hashlib.sha256(args.route.read_bytes()).hexdigest(),
        "road_lanelet_ids": [38,40,58,60] if not args.demo else None,
        "settings": cfg.__dict__,
        "obstacle_center_xy_m": list(obstacle),
        "same_feasible_graph_for_both_profiles": True,
        "states": sum(x["states"] for x in layers) + 1,
        "edges": sum(x["edges_from_previous"] for x in layers),
        "layers": layers,
        "safety_first": {"R9_clearance_shortfall_m_s": ca[0], "R5_corridor_offset_m2_s": ca[1], "R4_action_comfort_proxy": ca[2], "final_station_m": rows_a[-1]["s_m"]},
        "corridor_first": {"R9_clearance_shortfall_m_s": cb[0], "R5_corridor_offset_m2_s": cb[1], "R4_action_comfort_proxy": cb[2], "final_station_m": rows_b[-1]["s_m"]},
        "divergence": delta,
        "warnings": [
            "NOT original CommonRoad R9/R5 cost formulas; these are synthetic illustrative proxies.",
            "Comparison uses LEXICOGRAPHIC TOTAL PRIORITIES, not upstream partially ordered Rulebook dominance.",
            "Obstacle is synthetic and stationary; actor footprints are circles.",
            "No car steering/bicycle dynamics, polygon road boundaries, or certified continuous collision checking.",
            "Constant acceleration per step; quantized longitudinal position and lateral position.",
            "Corridor lateral bounds are ASSUMPTIONS, not verified Town01 lane widths.",
            "World coordinates based on the centerline tangent and lateral offset, with no yaw tracking error.",
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("\n===== MILESTONE 4B: MOTION GRAPH =====")
    print("Route:", summary["route_source"])
    print("Graph:", summary["states"], "time-layered states;", summary["edges"], "feasible transition edges")
    print("Obstacle (synthetic): center=", tuple(round(z, 3) for z in obstacle))
    for name, costs, rows in [("R9 first", ca, rows_a), ("R5 first", cb, rows_b)]:
        print(f"{name}: R9={costs[0]:.4f}; R5={costs[1]:.4f}; R4={costs[2]:.3f}; final_s={rows[-1]['s_m']:.1f}m")
    print("Aligned separation: max={:.3f} m; final={:.3f} m".format(
        delta["max_aligned_separation_m"], delta["final_separation_m"]))
    print("Max lateral-offset difference: {:.3f} m; final progress difference: {:.3f} m".format(
        delta["max_lateral_offset_difference_m"], delta["final_station_difference_m"]))
    print("Outputs:", args.out)
    if delta["max_aligned_separation_m"] < 1e-6:
        print("NO divergence for these settings; adjust synthetic conflict and evaluate again.")
    else:
        print("DIVERGENCE detected in simplified graph model (not CARLA execution).")


if __name__ == "__main__":
    main()
