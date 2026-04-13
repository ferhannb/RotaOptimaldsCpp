#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import tempfile

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def resolve_binary(explicit_bin):
    if explicit_bin:
        return explicit_bin
    for candidate in ("build_colreg_check/rota_optimal_ds", "build/rota_optimal_ds"):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("rota_optimal_ds binary not found; build the project or pass --bin")


def run_colreg_scan(binary_path, scenario_path, dt, steps, out_log):
    cmd = [
        binary_path,
        "--scenario",
        scenario_path,
        "--colreg-scan",
        "--scan-dt",
        str(dt),
        "--scan-steps",
        str(steps),
        "--out-colreg-log",
        out_log,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def load_scan_csv(path, target_id=None):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError("empty COLREG scan log")

    targets = sorted({row["target_id"] for row in rows})
    if target_id is None:
        if len(targets) > 1:
            raise ValueError(f"multiple targets present {targets}; pass --target")
        target_id = targets[0]

    rows = [row for row in rows if row["target_id"] == target_id]
    if not rows:
        raise ValueError(f"target_id {target_id} not found in COLREG scan log")

    data = {
        "target_id": target_id,
        "step": np.array([int(row["step"]) for row in rows], dtype=int),
        "time_s": np.array([float(row["time_s"]) for row in rows]),
        "own_x": np.array([float(row["own_x"]) for row in rows]),
        "own_y": np.array([float(row["own_y"]) for row in rows]),
        "own_course_deg": np.array([float(row["own_course_deg"]) for row in rows]),
        "target_x": np.array([float(row["target_x"]) for row in rows]),
        "target_y": np.array([float(row["target_y"]) for row in rows]),
        "target_course_deg": np.array([float(row["target_course_deg"]) for row in rows]),
        "type": [row["type"] for row in rows],
        "role": [row["role"] for row in rows],
        "geometry_type": [row.get("geometry_type", row["type"]) for row in rows],
        "geometry_role": [row.get("geometry_role", row["role"]) for row in rows],
        "risk": np.array([int(row["risk"]) for row in rows], dtype=int),
        "dcpa": np.array([float(row["dcpa"]) for row in rows]),
        "tcpa": np.array([float(row["tcpa"]) for row in rows]),
        "alpha0_deg": np.array([float(row["alpha0_deg"]) for row in rows]),
        "beta0_deg": np.array([float(row["beta0_deg"]) for row in rows]),
    }
    return data


def compute_limits(data):
    xs = np.concatenate([data["own_x"], data["target_x"]])
    ys = np.concatenate([data["own_y"], data["target_y"]])
    pad = max(2.0, 0.15 * max(xs.max() - xs.min(), ys.max() - ys.min(), 1.0))
    return (
        xs.min() - pad,
        xs.max() + pad,
        ys.min() - pad,
        ys.max() + pad,
    )


def trim_to_approach_phase(data):
    ranges = np.hypot(data["target_x"] - data["own_x"], data["target_y"] - data["own_y"])
    if ranges.size == 0:
        return data

    last_idx = int(np.argmin(ranges))
    trimmed = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            trimmed[key] = value[: last_idx + 1]
        elif isinstance(value, list):
            trimmed[key] = value[: last_idx + 1]
        else:
            trimmed[key] = value
    return trimmed


def build_animation(data, title, interval_ms):
    fig, (ax_xy, ax_metric) = plt.subplots(1, 2, figsize=(12.5, 6), constrained_layout=True)

    x_min, x_max, y_min, y_max = compute_limits(data)
    ax_xy.set_xlim(x_min, x_max)
    ax_xy.set_ylim(y_min, y_max)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_title(title)

    own_traj, = ax_xy.plot([], [], color="tab:blue", linewidth=2, label="own ship trail")
    target_traj, = ax_xy.plot([], [], color="tab:red", linewidth=2, label="target trail")
    own_pt, = ax_xy.plot([], [], marker="o", color="tab:blue", markersize=8, linestyle="None", label="own ship")
    target_pt, = ax_xy.plot([], [], marker="o", color="tab:red", markersize=8, linestyle="None", label=data["target_id"])
    cpa_line, = ax_xy.plot([], [], color="black", linewidth=1.2, linestyle=":", alpha=0.8, label="instant range")
    status_text = ax_xy.text(
        0.02,
        0.98,
        "",
        transform=ax_xy.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.96),
    )
    ax_xy.legend(loc="lower right")

    times = data["time_s"]
    ax_metric.plot(times, data["dcpa"], color="tab:green", label="DCPA")
    ax_metric.plot(times, data["tcpa"], color="tab:purple", label="TCPA")
    risk_fill = ax_metric.fill_between(
        times,
        0.0,
        1.0,
        where=data["risk"] > 0,
        transform=ax_metric.get_xaxis_transform(),
        color="#ffe0e0",
        alpha=0.6,
        label="risk window",
    )
    cursor = ax_metric.axvline(times[0], color="black", linestyle="--", linewidth=1.2)
    ax_metric.set_xlabel("time [s]")
    ax_metric.set_ylabel("metric value")
    ax_metric.grid(True, alpha=0.3)
    ax_metric.legend(loc="best")
    ax_metric.set_title("Classification Metrics")

    def frame_color(i):
        if data["risk"][i] > 0:
            return "#fff4e5"
        return "#eef7ee"

    def update(i):
        own_traj.set_data(data["own_x"][: i + 1], data["own_y"][: i + 1])
        target_traj.set_data(data["target_x"][: i + 1], data["target_y"][: i + 1])
        own_pt.set_data([data["own_x"][i]], [data["own_y"][i]])
        target_pt.set_data([data["target_x"][i]], [data["target_y"][i]])
        cpa_line.set_data(
            [data["own_x"][i], data["target_x"][i]],
            [data["own_y"][i], data["target_y"][i]],
        )
        status_text.set_text(
            "\n".join(
                [
                    f"t = {data['time_s'][i]:.2f} s",
                    f"tracked = {data['type'][i]}",
                    f"tracked role = {data['role'][i]}",
                    f"geometry = {data['geometry_type'][i]}",
                    f"geometry role = {data['geometry_role'][i]}",
                    f"risk active = {'yes' if data['risk'][i] else 'no'}",
                    f"DCPA = {data['dcpa'][i]:.2f}",
                    f"TCPA = {data['tcpa'][i]:.2f}",
                    f"alpha0 = {data['alpha0_deg'][i]:.1f} deg",
                    f"beta0 = {data['beta0_deg'][i]:.1f} deg",
                ]
            )
        )
        status_text.set_bbox(dict(boxstyle="round,pad=0.35", fc=frame_color(i), ec="0.6", alpha=0.96))
        cursor.set_xdata([data["time_s"][i], data["time_s"][i]])
        return own_traj, target_traj, own_pt, target_pt, cpa_line, status_text, cursor, risk_fill

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    return fig, ani


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, help="scenario ini path")
    parser.add_argument("--bin", default=None, help="path to rota_optimal_ds binary")
    parser.add_argument("--dt", type=float, default=0.5, help="scan time step")
    parser.add_argument("--steps", type=int, default=80, help="scan steps")
    parser.add_argument("--target", default=None, help="target_id when scan contains multiple targets")
    parser.add_argument("--log", default=None, help="existing COLREG scan csv; skips running C++ scan")
    parser.add_argument("--save", default=None, help="save animation to gif/html")
    parser.add_argument("--fps", type=int, default=6, help="save fps")
    parser.add_argument("--interval-ms", type=int, default=220, help="interactive frame interval in ms")
    parser.add_argument(
        "--full-window",
        action="store_true",
        help="show the full scan, including post-CPA separation",
    )
    parser.add_argument("--no-show", action="store_true", help="do not open interactive window")
    args = parser.parse_args()

    if args.log:
      log_path = args.log
    else:
      binary_path = resolve_binary(args.bin)
      tmp = tempfile.NamedTemporaryFile(prefix="colreg_scan_", suffix=".csv", delete=False)
      tmp.close()
      log_path = tmp.name
      run_colreg_scan(binary_path, args.scenario, args.dt, args.steps, log_path)

    data = load_scan_csv(log_path, args.target)
    if not args.full_window:
        data = trim_to_approach_phase(data)
    fig, ani = build_animation(
        data,
        f"COLREG Animation: {os.path.basename(args.scenario)}",
        args.interval_ms,
    )

    if args.save:
      if args.save.lower().endswith(".gif"):
        writer = animation.PillowWriter(fps=args.fps)
        ani.save(args.save, writer=writer)
      elif args.save.lower().endswith(".html"):
        ani.save(args.save, writer="html")
      else:
        raise ValueError("unsupported save format; use .gif or .html")

    if not args.no_show:
      plt.show()


if __name__ == "__main__":
    main()
