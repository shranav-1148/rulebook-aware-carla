"""Milestone 4A: inspect the graph implicit in the Town01 CommonRoad lanelets.

Input:   integration/maps/Town01.rpp_compat.xml
Outputs: integration/output/town01_graph/*

This is a ROAD TOPOLOGY / geometric waypoint experiment, not a motion
planner, not a CARLA controller, and not a Rulebook priority experiment.
"""

import argparse
from collections import deque
import csv
import hashlib
import heapq
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent
DEFAULT_MAP = BASE / "maps" / "Town01.rpp_compat.xml"
DEFAULT_OUTPUT = BASE / "output" / "town01_graph"
KNOWN_ROUTE = (38, 40, 58, 60)


def _center_points(lanelet):
    points = [(float(p[0]), float(p[1])) for p in lanelet.center_vertices]
    if len(points) < 2 or not all(math.isfinite(z) for point in points for z in point):
        raise ValueError(f"Invalid centerline for lanelet {lanelet.lanelet_id}")
    if polyline_length(points) <= 0.0:
        raise ValueError(f"Zero-length centerline for lanelet {lanelet.lanelet_id}")
    return points


def polyline_length(points):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(points, points[1:]))


def build_graph(lanelets):
    """Make graph data using only documented lanelet connectivity attributes.

    Returns (nodes, edges, missing_references).
    Edges labeled 'lateral_candidate' are NOT established legal lane changes.
    """
    by_id = {int(l.lanelet_id): l for l in lanelets}
    if len(by_id) != len(lanelets):
        raise ValueError("Duplicate lanelet IDs")
    nodes = {}
    for lanelet_id, lanelet in sorted(by_id.items()):
        points = _center_points(lanelet)
        nodes[lanelet_id] = {
            "lanelet_id": lanelet_id,
            "length_m": polyline_length(points),
            "center_x_m": sum(p[0] for p in points) / len(points),
            "center_y_m": sum(p[1] for p in points) / len(points),
            "centerline": points,
        }

    edges = []
    missing = []
    seen = set()

    def add(src, dst, kind, side=""):
        dst = int(dst)
        if dst not in nodes:
            missing.append({"source": src, "target": dst, "type": kind, "side": side})
            return
        key = (src, dst, kind, side)
        if key in seen:
            return
        seen.add(key)
        edges.append({
            "source": src,
            "target": dst,
            "type": kind,
            "side": side,
            "target_length_m": nodes[dst]["length_m"],
            "lane_change_legality_verified": False if kind == "lateral_candidate" else None,
        })

    for lanelet_id, lanelet in sorted(by_id.items()):
        for successor in sorted(set(lanelet.successor or [])):
            add(lanelet_id, successor, "successor")
        for side in ("left", "right"):
            other_id = getattr(lanelet, f"adj_{side}", None)
            same_direction = bool(getattr(lanelet, f"adj_{side}_same_direction", False))
            if other_id is not None and same_direction:
                add(lanelet_id, other_id, "lateral_candidate", side=side)

    edges.sort(key=lambda e: (e["source"], e["target"], e["type"], e["side"]))
    return nodes, edges, missing


def successor_adjacency(nodes, edges):
    result = {lanelet_id: [] for lanelet_id in nodes}
    for edge in edges:
        if edge["type"] == "successor":
            result[edge["source"]].append(edge["target"])
    for successors in result.values():
        successors.sort()
    return result


def validate_chain(adjacency, route):
    if not route or any(lanelet_id not in adjacency for lanelet_id in route):
        return False
    return all(b in adjacency[a] for a, b in zip(route, route[1:]))


def bfs_route(adjacency, start, goal):
    """Fewest successor edges, ignoring physical distance."""
    if start not in adjacency or goal not in adjacency:
        return None
    queue = deque([(start,)])
    discovered = {start}
    while queue:
        path = queue.popleft()
        if path[-1] == goal:
            return list(path)
        for nxt in adjacency[path[-1]]:
            if nxt not in discovered:
                discovered.add(nxt)
                queue.append(path + (nxt,))
    return None


def distance_weighted_route(adjacency, nodes, start, goal):
    """Toy Dijkstra: successor edge cost = entire destination-lanelet length.

    This is NOT a calibrated shortest route from the start/goal midpoints;
    it ignores portions of the first and final lanelets and lane changes.
    """
    if start not in adjacency or goal not in adjacency:
        return None, None
    frontier = [(0.0, (start,))]
    best = {start: 0.0}
    while frontier:
        cost, path = heapq.heappop(frontier)
        here = path[-1]
        if cost > best[here] + 1e-9:
            continue
        if here == goal:
            return list(path), cost
        for nxt in adjacency[here]:
            new_cost = cost + nodes[nxt]["length_m"]
            if new_cost < best.get(nxt, float("inf")) - 1e-9:
                best[nxt] = new_cost
                heapq.heappush(frontier, (new_cost, path + (nxt,)))
    return None, None


def sample_centerline(points, step_m):
    """Sample a polyline at approximate arclength intervals; retain endpoints."""
    if step_m <= 0 or not math.isfinite(step_m):
        raise ValueError("step_m must be finite and > 0")
    pts = [(float(a), float(b)) for a, b in points]
    pieces = []
    total = 0.0
    for a, b in zip(pts, pts[1:]):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        if length > 1e-12:
            pieces.append((total, total + length, a, b))
            total += length
    if total <= 0:
        raise ValueError("Centerline contains no nonzero segments")

    distances = [0.0]
    k = 1
    while k * step_m < total - 1e-8:
        distances.append(k * step_m)
        k += 1
    distances.append(total)

    sampled = []
    j = 0
    for s in distances:
        while j < len(pieces) - 1 and s > pieces[j][1] + 1e-9:
            j += 1
        lo, hi, a, b = pieces[j]
        ratio = min(1.0, max(0.0, (s - lo) / (hi - lo)))
        x = a[0] + ratio * (b[0] - a[0])
        y = a[1] + ratio * (b[1] - a[1])
        heading = math.atan2(b[1] - a[1], b[0] - a[0])
        sampled.append({"s_m": s, "x_m": x, "y_m": y, "heading_rad": heading})
    return sampled


def make_route_waypoint_graph(nodes, route, spacing_m):
    """Build a geometric, forward-only waypoint chain over a chosen route.

    Connector edges use lanelet successor topology but include physical gap
    distances as diagnostics. Neither speed nor obstacles are represented.
    """
    if not route or any(i not in nodes for i in route):
        raise ValueError("Unknown or empty route")
    points = []
    edges = []
    connector_gaps = []
    prev_end_id = None
    for lanelet_id in route:
        rows = sample_centerline(nodes[lanelet_id]["centerline"], spacing_m)
        lanelet_start_id = len(points)
        for row in rows:
            idx = len(points)
            points.append({"waypoint_id": idx, "lanelet_id": lanelet_id, **row})
            if idx > lanelet_start_id:
                prev = points[idx - 1]
                edges.append({
                    "source": idx - 1,
                    "target": idx,
                    "type": "along_centerline",
                    "length_m": math.hypot(row["x_m"] - prev["x_m"],
                                           row["y_m"] - prev["y_m"]),
                })
        if prev_end_id is not None:
            a, b = points[prev_end_id], points[lanelet_start_id]
            gap = math.hypot(a["x_m"] - b["x_m"], a["y_m"] - b["y_m"])
            connector_gaps.append({"from_lanelet": a["lanelet_id"],
                                   "to_lanelet": lanelet_id, "gap_m": gap})
            edges.append({"source": prev_end_id, "target": lanelet_start_id,
                          "type": "successor_connector", "length_m": gap})
        prev_end_id = len(points) - 1
    return points, edges, connector_gaps


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def plot_town01(nodes, route, waypoint_nodes, path):
    """Optional static figure: actual lanelet centerlines, not a state-space graph."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("PLOT SKIPPED: matplotlib isn't installed; graph data was exported.")
        return False

    fig, ax = plt.subplots(figsize=(10, 8))
    for info in nodes.values():
        x, y = zip(*info["centerline"])
        ax.plot(x, y, color="0.72", linewidth=0.9, zorder=1)
    for i in route:
        x, y = zip(*nodes[i]["centerline"])
        ax.plot(x, y, color="tab:blue", linewidth=3.3, zorder=3)
        c = nodes[i]["centerline"][len(nodes[i]["centerline"]) // 2]
        ax.annotate(str(i), c, fontsize=8, color="tab:blue", zorder=5)
    if 51 in nodes:
        c = nodes[51]["centerline"][len(nodes[51]["centerline"]) // 2]
        ax.annotate("51", c, color="tab:orange", fontsize=9, zorder=6)
    if waypoint_nodes:
        ax.plot([p["x_m"] for p in waypoint_nodes],
                [p["y_m"] for p in waypoint_nodes], ".", color="tab:red",
                markersize=2.0, zorder=4)
    ax.set_title("Town01 lanelet centerlines and successor route (not vehicle motions)")
    ax.set_xlabel("CommonRoad X (m)")
    ax.set_ylabel("CommonRoad Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(str(path), dpi=160)
    plt.close(fig)
    return True



def plot_local_successor_graph(nodes, edges, route, path):
    """Topology diagram of the route and its outgoing alternatives.

    Diagram layout is abstract; positions do NOT represent geographic distances.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    route = tuple(route)
    focus = set(route)
    outgoing = {i: [] for i in route}
    for edge in edges:
        if edge["type"] == "successor" and edge["source"] in route and edge["target"] not in focus:
            focus.add(edge["target"])
            outgoing[edge["source"]].append(edge["target"])

    pos = {node: (float(i), 0.0) for i, node in enumerate(route)}
    for i, src in enumerate(route):
        extras = sorted(set(outgoing[src]))
        for j, target in enumerate(extras):
            if target not in pos:
                pos[target] = (float(i) + 0.55, 0.85 + j * 0.70)

    fig, ax = plt.subplots(figsize=(10, 4))
    for edge in edges:
        src, dst = edge["source"], edge["target"]
        if edge["type"] != "successor" or src not in focus or dst not in focus:
            continue
        color = "tab:blue" if src in route and dst in route and route.index(dst) == route.index(src) + 1 else "0.6"
        ax.annotate("", xy=pos[dst], xytext=pos[src],
                    arrowprops={"arrowstyle": "->", "color": color,
                                "lw": 1.8, "shrinkA": 20, "shrinkB": 20}, zorder=1)
    for lanelet_id in sorted(focus):
        x, y = pos[lanelet_id]
        ax.text(x, y, str(lanelet_id), ha="center", va="center", fontsize=11,
                color="white" if lanelet_id in route else "black",
                bbox={"boxstyle": "circle,pad=0.55", "facecolor": "tab:blue" if lanelet_id in route else "0.85",
                      "edgecolor": "0.45", "linewidth": 1.2}, zorder=3)
    ax.set_xlim(-0.45, len(route) - 0.3)
    ax.set_ylim(-0.5, max((p[1] for p in pos.values()), default=0) + 0.7)
    ax.set_title("Directed successor graph near the Town01 route")
    ax.text(0.5, -0.07, "Abstract topology (not geographic distance); blue nodes = verified route",
            transform=ax.transAxes, ha="center", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(path), dpi=160)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--spacing-m", type=float, default=5.0,
                        help="Centerline waypoint spacing in meters (default: 5)")
    args = parser.parse_args()

    if not args.map.is_file():
        raise FileNotFoundError(f"CommonRoad Town01 map missing: {args.map}")

    from commonroad.common.file_reader import CommonRoadFileReader
    scenario, _ = CommonRoadFileReader(str(args.map)).open()
    nodes, edges, missing = build_graph(scenario.lanelet_network.lanelets)
    adjacency = successor_adjacency(nodes, edges)
    if not validate_chain(adjacency, KNOWN_ROUTE):
        raise RuntimeError("Known route 38->40->58->60 isn't a successor chain in this map")

    bfs = bfs_route(adjacency, KNOWN_ROUTE[0], KNOWN_ROUTE[-1])
    dij, dij_cost = distance_weighted_route(
        adjacency, nodes, KNOWN_ROUTE[0], KNOWN_ROUTE[-1])
    waypoints, waypoint_edges, connector_gaps = make_route_waypoint_graph(
        nodes, KNOWN_ROUTE, args.spacing_m)

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "lanelet_nodes.csv",
              ["lanelet_id", "length_m", "center_x_m", "center_y_m"],
              [nodes[i] for i in sorted(nodes)])
    write_csv(args.out / "lanelet_edges.csv",
              ["source", "target", "type", "side", "target_length_m",
               "lane_change_legality_verified"], edges)
    write_json(args.out / "successor_adjacency.json", adjacency)
    write_csv(args.out / "route_waypoints.csv",
              ["waypoint_id", "lanelet_id", "s_m", "x_m", "y_m", "heading_rad"],
              waypoints)
    write_csv(args.out / "route_waypoint_edges.csv",
              ["source", "target", "type", "length_m"], waypoint_edges)
    write_json(args.out / "missing_lanelet_references.json", missing)

    plot_saved = plot_town01(nodes, KNOWN_ROUTE, waypoints,
                             args.out / "town01_lanelet_map.png")
    topo_plot_saved = plot_local_successor_graph(
        nodes, edges, KNOWN_ROUTE, args.out / "town01_successor_graph.png")
    successor_count = sum(e["type"] == "successor" for e in edges)
    lateral_count = sum(e["type"] == "lateral_candidate" for e in edges)
    dead_ends = sum(not adjacency[i] for i in nodes)
    lanelet_51 = {
        "present": 51 in nodes,
        "successors": adjacency.get(51),
        "same_direction_lateral_neighbors": [
            {"target": e["target"], "side": e["side"]}
            for e in edges if e["source"] == 51 and e["type"] == "lateral_candidate"
        ],
    }
    summary = {
        "map_input": str(args.map),
        "map_sha256": hashlib.sha256(args.map.read_bytes()).hexdigest(),
        "lanelet_nodes": len(nodes),
        "successor_edges": successor_count,
        "same_direction_lateral_adjacency_candidates": lateral_count,
        "lateral_legality_checked": False,
        "lanelets_without_successors": dead_ends,
        "missing_references": len(missing),
        "verified_chain": list(KNOWN_ROUTE),
        "known_chain_length_m_full_lanelets": sum(nodes[i]["length_m"] for i in KNOWN_ROUTE),
        "bfs_min_hop_path_38_to_60": bfs,
        "dijkstra_target_lanelet_length_path_38_to_60": dij,
        "dijkstra_cost_m_excluding_start_lanelet": dij_cost,
        "lanelet_51": lanelet_51,
        "route_spacing_m": args.spacing_m,
        "route_waypoints": len(waypoints),
        "route_waypoint_edges": len(waypoint_edges),
        "successor_connector_gaps": connector_gaps,
        "geographic_plot_saved": plot_saved,
        "topology_plot_saved": topo_plot_saved,
        "notes": [
            "Road topology only; candidate lane changes are NOT legally verified.",
            "Road connection is not a vehicle trajectory or a motion primitive.",
            "Waypoint edges have distances but no time, speed, curvature, obstacle or Rulebook cost.",
            "Known compatibility map replaced 'curb' labels with 'unknown' in an earlier step.",
            "BFS and Dijkstra are illustrative; the installed RoutePlanner is independent.",
        ],
    }
    write_json(args.out / "graph_summary.json", summary)

    print("\n===== TOWN01 LANELET GRAPH EXPERIMENT =====")
    print("Lanelet nodes:", len(nodes))
    print("Directed successor edges:", successor_count)
    print("Same-direction lateral-adjacency candidates (NOT legal lane changes):", lateral_count)
    print("Lanelets with no successors:", dead_ends)
    print("Known route confirmed:", list(KNOWN_ROUTE))
    print("BFS (min hops) 38 -> 60:", bfs)
    print("Dijkstra (target lanelet lengths) 38 -> 60:", dij)
    print("Lanelet 51:", lanelet_51)
    print(f"Waypoint spacing: {args.spacing_m:g} m; waypoint nodes: {len(waypoints)}")
    print("Connector gaps (m):", [round(g["gap_m"], 3) for g in connector_gaps])
    if any(g["gap_m"] > 2.0 for g in connector_gaps):
        print("WARNING: at least one successor-centerline endpoint gap exceeds 2 m; review geometry")
    if missing:
        print("WARNING: references to lanelet IDs absent from loaded network:", len(missing))
    print("Outputs:", args.out)
    print("SUCCESS: extracted road topology and geometric waypoints; no Rulebook planning run.")


if __name__ == "__main__":
    main()
