#!/usr/bin/env python3
"""Recreate the five offline CARLA/CommonRoad proxy route figures with Matplotlib.

This script does not connect to CARLA. It reconstructs the selected trajectories
from the rear-axle initial state and the acceleration / steering-rate actions
saved in the four experiment JSON reports. The integration follows the
five-state kinematic bicycle model and RK4 convention in town01_bicycle_graph.py.

Install: python -m pip install numpy matplotlib
Run:     python plot_commonroad_routes.py
Custom:  python plot_commonroad_routes.py --data-dir PATH --out PATH --format both

Default inputs (under ./reports, next to this script):
  preview_2s.json        Original: report.json
  settled_6s.json        Original: report(2).json
  encounter_beam200.json Original: report(3).json
  encounter_beam400.json Original: report(4).json

Authoritative for trajectory shapes: saved actions in each JSON report.
The original lanelet map/route CSV is NOT part of these inputs. Therefore the
horizontal line is the initial heading, NOT the CommonRoad route centerline.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # export PNG / PDF without a GUI or CARLA window
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

BLUE = "#577D99"       # muted steel blue / safety-first
SAND = "#C4A176"       # sandstone / corridor-first
BLUE_LIGHT = "#819BB2" # beam-400 safety-first
CONE = "#C17F55"
INK = "#273745"
GRAY = "#64717D"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.edgecolor": "#D9DEDF", "axes.labelcolor": "#394D5D",
    "xtick.color": "#5C6872", "ytick.color": "#5C6872",
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def derivative(state, acceleration, steering_rate_rad_s, wheelbase):
    """State = (rear_x, rear_y, yaw_rad, speed_mps, wheel_angle_rad)."""
    x, y, yaw, speed, wheel_angle = state
    return (
        speed * math.cos(yaw),
        speed * math.sin(yaw),
        speed * math.tan(wheel_angle) / wheelbase,
        acceleration,
        steering_rate_rad_s,
    )


def rk4_step(state, acceleration, steering_rate_deg_s, dt, wheelbase):
    """4th-order Runge–Kutta, matching town01_bicycle_graph.py."""
    steering_rate = math.radians(steering_rate_deg_s)
    k1 = derivative(state, acceleration, steering_rate, wheelbase)
    s2 = tuple(q + 0.5 * dt * k for q, k in zip(state, k1))
    k2 = derivative(s2, acceleration, steering_rate, wheelbase)
    s3 = tuple(q + 0.5 * dt * k for q, k in zip(state, k2))
    k3 = derivative(s3, acceleration, steering_rate, wheelbase)
    s4 = tuple(q + dt * k for q, k in zip(state, k3))
    k4 = derivative(s4, acceleration, steering_rate, wheelbase)
    nxt = [q + dt*(a+2*b+2*c+d)/6 for q,a,b,c,d in zip(state,k1,k2,k3,k4)]
    nxt[2] = wrap_angle(nxt[2])
    return tuple(nxt)


def reconstruct(report, profile):
    """Return arrays (time_s, body_center_CR_xy_m) for selected actions."""
    cfg = report["settings"]
    dt = float(cfg["dt_s"])
    internal_dt = float(cfg["integration_dt_s"])
    wheelbase = float(cfg["wheelbase_m"])
    if dt <= 0 or internal_dt <= 0 or wheelbase <= 0:
        raise ValueError("Invalid dt, integration timestep, or wheelbase in report")
    q = tuple(float(report["start_state"][key]) for key in
              ("x", "y", "yaw", "v", "steer"))
    subdivisions = max(2, math.ceil(dt/internal_dt))
    h = dt/subdivisions

    def body_center(s):
        # Original planner stores its bicycle position at the rear axle.
        return (s[0] + 0.5*wheelbase*math.cos(s[2]),
                s[1] + 0.5*wheelbase*math.sin(s[2]))

    positions = [body_center(q)]
    times = [0.0]
    for step, action in enumerate(report["profiles"][profile]["actions"]):
        a = float(action["acceleration_mps2"])
        rate = float(action["steering_rate_deg_s"])
        for substep in range(1, subdivisions+1):
            q = rk4_step(q, a, rate, h, wheelbase)
            positions.append(body_center(q))
            times.append(step*dt + substep*h)
    return np.asarray(times), np.asarray(positions)


def validate_report(report, label):
    """Catch incorrect/mismatched reports before making misleading plots."""
    ta, a = reconstruct(report, "safety_first")
    tb, b = reconstruct(report, "corridor_first")
    if not np.allclose(ta, tb, atol=1e-10):
        raise ValueError(f"{label}: trajectory timestamps differ")
    separation = np.linalg.norm(a-b, axis=1)
    observed_max = float(np.max(separation))
    observed_final = float(separation[-1])
    expected = report["divergence"]
    expected_max = float(expected["max_time_aligned_separation_m"])
    expected_final = float(expected["final_separation_m"])
    if not (np.isclose(observed_max, expected_max, atol=1e-5) and
            np.isclose(observed_final, expected_final, atol=1e-5)):
        raise ValueError(
            f"{label}: reconstructed separations differ from archived report. "
            f"max {observed_max:.6f} vs {expected_max:.6f}, "
            f"final {observed_final:.6f} vs {expected_final:.6f}. "
            "Check whether this is the correct report."
        )
    print(f"  {label:>16}: {len(ta):3d} points, "
          f"max separation {observed_max:7.4f} m, "
          f"final separation {observed_final:7.4f} m [verified]")


class PlotFrame:
    """Convert CommonRoad XY positions to a forward/lateral plotting frame."""
    def __init__(self, reference_report):
        self.reference = reference_report
        start = reference_report["start_state"]
        cfg = reference_report["settings"]
        theta = float(start["yaw"])
        wb = float(cfg["wheelbase_m"])
        self.origin = np.array([
            start["x"]+0.5*wb*math.cos(theta),
            start["y"]+0.5*wb*math.sin(theta),
        ])
        self.fwd = np.array([math.cos(theta), math.sin(theta)])
        self.left = np.array([-math.sin(theta), math.cos(theta)])

    def local(self, coords):
        diff = np.asarray(coords)-self.origin
        return diff @ self.fwd, diff @ self.left

    def profile(self, report, name):
        t, body_xy = reconstruct(report, name)
        x, y = self.local(body_xy)
        return t, x, y

    def cone(self, ax, label=True):
        # Measured CARLA cone AABB, transformed to CommonRoad XY, from the report.
        footprint = np.asarray(self.reference["measured_obstacle_bbox_cr_xy_exterior_m"])
        if np.allclose(footprint[0], footprint[-1]):
            footprint = footprint[:-1]
        x, y = self.local(footprint)
        ax.add_patch(Polygon(np.column_stack((x,y)), closed=True,
                             facecolor=CONE, edgecolor="#B26F47",
                             linewidth=1.2, alpha=0.9, zorder=6))
        if label:
            ax.annotate("Measured cone footprint", (np.mean(x), np.mean(y)),
                        xytext=(24.0,-3.1), fontsize=9, color="#7D5D4C",
                        arrowprops={"arrowstyle":"-", "color":"#B99C87"},
                        ha="center")


def style(ax, title, xlim=(-1, 31), ylim=(-3.8, 2.1)):
    ax.set(xlim=xlim, ylim=ylim,
           xlabel="Forward distance from the starting car (m)",
           ylabel="Lateral distance (m)")
    ax.set_title(title, loc="left", pad=12)
    ax.grid(color="#E5EBED", linewidth=0.7, alpha=0.75)
    ax.axhline(0, color="#B5BFBE", linewidth=1, ls=(0,(4,5)), alpha=0.9)
    ax.set_facecolor("#FBFCFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def stamp(ax, message):
    ax.text(.99,.975,message,transform=ax.transAxes,ha="right",va="top",
            color="#61707A",fontsize=8,
            bbox={"facecolor":"white", "edgecolor":"#E2E7E7",
                  "boxstyle":"round,pad=.35", "alpha":.95})


def draw_start(ax):
    ax.scatter([0], [0], s=85, marker="s", facecolor=INK, zorder=8)


def draw_two_paths(ax, frame, report, legend=True, line_width=3):
    for profile, color, label in (
        ("safety_first", BLUE, "Safety-first"),
        ("corridor_first", SAND, "Corridor-first"),
    ):
        _, x, y = frame.profile(report, profile)
        ax.plot(x, y, color=color, lw=line_width,
                solid_capstyle="round", zorder=4, label=label)
        ax.scatter([x[-1]], [y[-1]], s=48, facecolor="white",
                   edgecolor=color, linewidths=2, zorder=9)
    if legend:
        ax.legend(loc="upper left", frameon=True, edgecolor="#E1E6E7",
                  facecolor="white", fontsize=9, ncol=2)


def save_figure(fig, output_path, name, fmt, dpi):
    for suffix in (["png", "pdf"] if fmt == "both" else [fmt]):
        dest = output_path / f"{name}.{suffix}"
        fig.savefig(dest, dpi=dpi, bbox_inches="tight", facecolor="white")
        print("  saved", dest)
    plt.close(fig)


def make_plots(reports, output_path, fmt, dpi):
    frame = PlotFrame(reports["beam200"])
    output_path.mkdir(parents=True, exist_ok=True)

    # 1 — Early horizons: no difference between profiles.
    fig,ax=plt.subplots(figsize=(11.3,4.65))
    style(ax,"The first two route tests: both preferences use the same line",
          (-0.6,26),(-3.65,1.95))
    frame.cone(ax)
    _,x,y=frame.profile(reports["settled"],"safety_first")
    ax.plot(x,y,lw=3.5,color=BLUE,label="Stable 6-second plan (both profiles overlap)")
    ax.scatter([x[-1]],[y[-1]],s=65,facecolor="white",edgecolor=BLUE,lw=2,zorder=8)
    _,xs,ys=frame.profile(reports["preview"],"safety_first")
    ax.plot(xs,ys,lw=4,color=SAND,label="Early 2-second preview (unsettled observation)")
    ax.scatter([xs[-1]],[ys[-1]],s=60,facecolor="white",edgecolor=SAND,lw=2,zorder=8)
    ax.annotate("2 s ends here",(xs[-1],ys[-1]),xytext=(4.3,1.05),
                fontsize=10,color=GRAY,arrowprops={"arrowstyle":"->","color":"#829099"})
    ax.annotate("6 s ends here",(x[-1],y[-1]),xytext=(10.7,1.05),
                fontsize=10,color=GRAY,arrowprops={"arrowstyle":"->","color":"#829099"})
    draw_start(ax)
    ax.legend(loc="lower left",fontsize=8.8,frameon=True,edgecolor="#E6E9EB")
    stamp(ax,"Reconstructed from archived actions; map lane boundaries not shown")
    fig.tight_layout()
    save_figure(fig,output_path,"route_01_short_vs_six",fmt,dpi)

    # 2 — The 8-second, beam-200 obstacle encounter.
    fig,ax=plt.subplots(figsize=(11.3,5.1))
    style(ax,"Eight seconds, beam 200: two trajectories become visible",
          (-0.6,31),(-3.8,2.2))
    frame.cone(ax)
    draw_two_paths(ax,frame,reports["beam200"])
    t,xa,ya=frame.profile(reports["beam200"],"safety_first")
    _,xb,yb=frame.profile(reports["beam200"],"corridor_first")
    sep=np.hypot(xa-xb,ya-yb)
    idx=int(np.argmax(sep))
    ax.plot([xa[idx],xb[idx]],[ya[idx],yb[idx]],color="#7D868C",
            ls=":",lw=1.8,zorder=2)
    ax.annotate(f"Max. time-aligned separation: {sep[idx]:.2f} m\n(at t = {t[idx]:.1f} s)",
                ((xa[idx]+xb[idx])/2,(ya[idx]+yb[idx])/2),
                xytext=(14,1.45),fontsize=9,color=INK,
                arrowprops={"arrowstyle":"-","color":"#88959C"})
    ax.annotate("Safety-first ends ~28 m",(xa[-1],ya[-1]),xytext=(24,1.75),
                fontsize=9,color=BLUE,arrowprops={"arrowstyle":"-","color":BLUE})
    ax.annotate("Corridor-first ends ~24.5 m",(xb[-1],yb[-1]),xytext=(23,-1.55),
                fontsize=9,color="#896A47",arrowprops={"arrowstyle":"-","color":SAND})
    draw_start(ax)
    fig.tight_layout()
    save_figure(fig,output_path,"route_02_eight_second_beam200",fmt,dpi)

    # 3 — Zoom near the obstacle.
    fig,ax=plt.subplots(figsize=(11.3,4.55))
    style(ax,"A closer look: clearance versus staying on the initial route line",
          (14.5,30.4),(-3.1,1.8))
    frame.cone(ax)
    draw_two_paths(ax,frame,reports["beam200"],legend=False,line_width=3.25)
    ax.text(25.0,.95,"Safety-first bends away\nfrom the cone",color=BLUE,fontsize=9)
    ax.text(20.9,-.65,"Corridor-first stays nearly straight",
            color="#896A47",fontsize=9)
    stamp(ax,"Initial-heading guide is NOT a lanelet centerline")
    fig.tight_layout()
    save_figure(fig,output_path,"route_03_cone_closeup",fmt,dpi)

    # 4 — Beam 200 and 400 on identical axes.
    fig,(top,bottom)=plt.subplots(2,1,figsize=(11.25,8.1),sharex=True,sharey=True)
    for ax,beam in ((top,200),(bottom,400)):
        report=reports[f"beam{beam}"]
        style(ax,f"Search beam {beam}: same start, cone and 8-second horizon",
              (-0.6,31),(-3.8,2.05))
        frame.cone(ax,label=False)
        draw_two_paths(ax,frame,report,legend=(beam==200))
        draw_start(ax)
        ax.text(.98,.11,f"Max. separation: {report['divergence']['max_time_aligned_separation_m']:.2f} m",
                transform=ax.transAxes,ha="right",fontsize=9,color="#506171",
                bbox={"facecolor":"white","edgecolor":"#E4E8EA","boxstyle":"round,pad=.3"})
    fig.tight_layout(h_pad=1.3)
    save_figure(fig,output_path,"route_04_beam200_vs_400",fmt,dpi)

    # 5 — Overlay selected curves from both search widths.
    fig,ax=plt.subplots(figsize=(11.3,4.75))
    style(ax,"Increasing the beam changes the selected clearance-first path",
          (-0.6,31),(-3.6,2.1))
    frame.cone(ax)
    for report,profile,color,label,ls,lw in (
        (reports["beam200"],"safety_first",BLUE,"Safety-first · beam 200","-",3),
        (reports["beam400"],"safety_first",BLUE_LIGHT,"Safety-first · beam 400","--",2.6),
        (reports["beam200"],"corridor_first",SAND,"Corridor-first · same in both","-",3),
    ):
        _,x,y=frame.profile(report,profile)
        ax.plot(x,y,color=color,ls=ls,lw=lw,label=label,zorder=3)
        ax.scatter([x[-1]],[y[-1]],s=50,edgecolor=color,
                   facecolor="white",lw=2,zorder=8)
    draw_start(ax)
    ax.legend(loc="upper left",fontsize=9,frameon=True,edgecolor="#E1E6E7")
    stamp(ax,"Selected proxy plans; not guaranteed globally optimal")
    fig.tight_layout()
    save_figure(fig,output_path,"route_05_beam_choice_overlay",fmt,dpi)


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    here=Path(__file__).resolve().parent
    ap.add_argument("--data-dir",type=Path,default=here/"reports",
                    help="Folder containing the four JSON report copies")
    ap.add_argument("--preview",type=Path,help="Override the 2-second preview report")
    ap.add_argument("--settled",type=Path,help="Override the stable 6-second report")
    ap.add_argument("--beam200",type=Path,help="Override the 8-second beam-200 report")
    ap.add_argument("--beam400",type=Path,help="Override the 8-second beam-400 report")
    ap.add_argument("--out",type=Path,default=here/"route_figures",
                    help="Output directory (default: ./route_figures)")
    ap.add_argument("--format",choices=["png","pdf","both"],default="png")
    ap.add_argument("--dpi",type=int,default=190)
    args=ap.parse_args()
    if args.dpi < 60 or args.dpi > 600:
        ap.error("--dpi must be between 60 and 600")
    files={
        "preview": args.preview or args.data_dir/"preview_2s.json",
        "settled": args.settled or args.data_dir/"settled_6s.json",
        "beam200": args.beam200 or args.data_dir/"encounter_beam200.json",
        "beam400": args.beam400 or args.data_dir/"encounter_beam400.json",
    }
    reports={}
    print("Loading and validating archived trajectories:")
    for key,path in files.items():
        if not path.is_file():
            ap.error(f"Missing {key} report: {path}\nUse --{key} PATH or --data-dir PATH")
        report=json.loads(path.read_text(encoding="utf-8"))
        validate_report(report,key)
        reports[key]=report
    make_plots(reports,args.out,args.format,args.dpi)
    print("Done. These are OFFLINE planned paths, not CARLA vehicle replays.")


if __name__=="__main__":
    main()
