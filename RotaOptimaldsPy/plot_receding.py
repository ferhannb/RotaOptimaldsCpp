#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def load_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def parse_bool(s):
    v = s.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"invalid bool value: {s}")


def parse_scenario_obstacles(path):
    obstacles = []
    clearance = 0.0
    if not path:
        return obstacles, clearance

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, value = [x.strip() for x in line.split("=", 1)]
            key = key.lower()
            if key in ("obstacle", "circle_obstacle"):
                toks = [t.strip() for t in value.split(",")]
                if len(toks) < 3:
                    continue
                cx = float(toks[0])
                cy = float(toks[1])
                r = float(toks[2])
                enabled = True
                if len(toks) > 3 and toks[3]:
                    enabled = parse_bool(toks[3])
                obstacles.append((cx, cy, r, enabled))
            elif key == "obstacle_clearance":
                clearance = float(value)

    return obstacles, clearance


def unique_detour_points(detour_x, detour_y, detour_obs):
    out = []
    seen = set()
    for x, y, obs in zip(detour_x, detour_y, detour_obs):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        key = (int(obs), round(float(x), 6), round(float(y), 6))
        if key in seen:
            continue
        seen.add(key)
        out.append((int(obs), float(x), float(y)))
    return out


def prepare_log_series(log):
    traj = np.column_stack((log["x"], log["y"]))
    psi_arr = np.arctan2(np.sin(log["psi"]), np.cos(log["psi"]))
    K_arr = log["K"]
    Kcmd_arr = log["Kcmd"][1:]
    ds_arr = log["ds"][1:]
    idx_arr = log["wp_index"].astype(int)

    names = log.dtype.names if log.dtype.names is not None else ()
    detour_pts = []
    if "detour_wp_x" in names and "detour_wp_y" in names and "detour_obs_idx" in names:
        detour_pts = unique_detour_points(
            log["detour_wp_x"],
            log["detour_wp_y"],
            log["detour_obs_idx"].astype(int),
        )

    s_grid = np.concatenate(([0.0], np.cumsum(ds_arr)))
    s_cmd = s_grid[:-1]

    wp_start_state = {}
    wp_start_step = {}
    if idx_arr.size > 0:
        idx_steps = idx_arr[1:] if idx_arr.size > 1 else np.array([], dtype=int)
        max_wp = int(np.max(idx_arr)) if idx_arr.size > 0 else -1
        for i in range(max_wp + 1):
            hits_state = np.where(idx_arr == i)[0]
            if hits_state.size > 0:
                wp_start_state[i] = int(hits_state[0])
            hits_step = np.where(idx_steps == i)[0]
            if hits_step.size > 0:
                wp_start_step[i] = int(hits_step[0])

    return {
        "traj": traj,
        "psi_arr": psi_arr,
        "K_arr": K_arr,
        "Kcmd_arr": Kcmd_arr,
        "ds_arr": ds_arr,
        "idx_arr": idx_arr,
        "detour_pts": detour_pts,
        "s_grid": s_grid,
        "s_cmd": s_cmd,
        "wp_start_state": wp_start_state,
        "wp_start_step": wp_start_step,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="receding_log.csv")
    p.add_argument("--log-overlay", default=None, help="second log to overlay on the same plots")
    p.add_argument("--wp", default="waypoints.csv")
    p.add_argument("--scenario", default=None, help="scenario ini path (for obstacle circles)")
    p.add_argument("--label", default=None, help="legend label for the primary log")
    p.add_argument("--overlay-label", default=None, help="legend label for the overlay log")
    p.add_argument("--save", default=None, help="save plot to an image file")
    p.add_argument("--no-show", action="store_true", help="do not open an interactive plot window")
    args = p.parse_args()

    log = load_csv(args.log)
    log_overlay = load_csv(args.log_overlay) if args.log_overlay else None
    wp = load_csv(args.wp)
    obstacles, clearance = parse_scenario_obstacles(args.scenario)

    main_series = prepare_log_series(log)
    overlay_series = prepare_log_series(log_overlay) if log_overlay is not None else None
    primary_label = args.label if args.label else os.path.splitext(os.path.basename(args.log))[0]
    overlay_label = None
    if args.log_overlay:
        overlay_label = (
            args.overlay_label
            if args.overlay_label
            else os.path.splitext(os.path.basename(args.log_overlay))[0]
        )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    ax = axes[0, 0]
    traj = main_series["traj"]
    ax.plot(traj[:, 0], traj[:, 1], marker="o", markersize=3.5, linewidth=1.8, label=primary_label, color="tab:blue")
    ax.scatter([traj[0, 0], traj[-1, 0]], [traj[0, 1], traj[-1, 1]], marker="x", color="tab:blue")
    if overlay_series is not None:
        traj_overlay = overlay_series["traj"]
        ax.plot(
            traj_overlay[:, 0],
            traj_overlay[:, 1],
            marker="s",
            markersize=3.0,
            linewidth=1.6,
            alpha=0.85,
            label=overlay_label,
            color="tab:red",
        )
        ax.scatter(
            [traj_overlay[0, 0], traj_overlay[-1, 0]],
            [traj_overlay[0, 1], traj_overlay[-1, 1]],
            marker="x",
            color="tab:red",
        )
    ax.scatter(wp["X"], wp["Y"], marker="*", s=120)
    for i, (cx, cy, r, enabled) in enumerate(obstacles):
        if not enabled:
            continue
        c = Circle((cx, cy), r, fill=False, linestyle="-", linewidth=2, edgecolor="crimson", alpha=0.9)
        ax.add_patch(c)
        if clearance > 0.0:
            c_clr = Circle(
                (cx, cy),
                r + clearance,
                fill=False,
                linestyle="--",
                linewidth=1.2,
                edgecolor="crimson",
                alpha=0.55,
            )
            ax.add_patch(c_clr)
        ax.scatter([cx], [cy], marker="+", c="crimson", s=80)
        ax.annotate(
            f"OBS{i}",
            (cx, cy),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color="crimson",
        )

    detour_labeled = False
    candidate_labeled = False
    for k, (obs_idx, dx, dy) in enumerate(main_series["detour_pts"]):
        ax.scatter(
            [dx],
            [dy],
            marker="D",
            s=75,
            c="tab:orange",
            label="selected detour WP" if not detour_labeled else None,
        )
        detour_labeled = True
        ax.annotate(
            f"D{k}",
            (dx, dy),
            xytext=(6, -10),
            textcoords="offset points",
            fontsize=8,
            color="tab:orange",
            fontweight="bold",
        )

        if 0 <= obs_idx < len(obstacles):
            cx, cy, _, enabled = obstacles[obs_idx]
            if enabled:
                ox = 2.0 * cx - dx
                oy = 2.0 * cy - dy
                ax.scatter(
                    [ox],
                    [oy],
                    marker="D",
                    s=70,
                    facecolors="none",
                    edgecolors="tab:orange",
                    linewidths=1.5,
                    label="other candidate WP" if not candidate_labeled else None,
                )
                candidate_labeled = True

    for i in range(len(wp)):
        seg = main_series["idx_arr"] == i
        if seg.any():
            seg_traj = traj[seg]
            n_pts = int(np.count_nonzero(seg))
            ax.plot(seg_traj[:, 0], seg_traj[:, 1], linewidth=2, alpha=0.5, color="tab:blue")
            mid = seg_traj[n_pts // 2]
            ax.annotate(
                f"WP{i} n={n_pts}",
                (mid[0], mid[1]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
            )
            if i in main_series["wp_start_state"]:
                k0 = main_series["wp_start_state"][i]
                ax.annotate(
                    f"WP{i}",
                    (traj[k0, 0], traj[k0, 1]),
                    xytext=(-14, -12),
                    textcoords="offset points",
                    fontsize=9,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#fff3bf", ec="gray", alpha=0.95),
                )
    ax.axis("equal")
    ax.grid(True)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("XY trajectory (Receding Horizon)")
    if obstacles or main_series["detour_pts"] or overlay_series is not None:
        ax.legend(loc="best")

    ax = axes[0, 1]
    ax.plot(main_series["s_grid"], main_series["K_arr"], marker="o", markersize=3.5, label=f"{primary_label} K", color="tab:blue")
    ax.plot(main_series["s_cmd"], main_series["Kcmd_arr"], "x--", label=f"{primary_label} K_cmd", color="tab:blue", alpha=0.8)
    if overlay_series is not None:
        ax.plot(
            overlay_series["s_grid"],
            overlay_series["K_arr"],
            marker="s",
            markersize=3.0,
            label=f"{overlay_label} K",
            color="tab:red",
        )
        ax.plot(
            overlay_series["s_cmd"],
            overlay_series["Kcmd_arr"],
            "--",
            label=f"{overlay_label} K_cmd",
            color="tab:red",
            alpha=0.8,
        )
    for i, k0 in main_series["wp_start_state"].items():
        if 0 <= k0 < len(main_series["K_arr"]):
            ax.annotate(
                f"WP{i}",
                (main_series["s_grid"][k0], main_series["K_arr"][k0]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
            )
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("s [m]")
    ax.set_ylabel("K [1/m]")
    ax.set_title("Curvature profile")

    ax = axes[1, 0]
    ax.plot(main_series["s_grid"], main_series["psi_arr"], marker="o", markersize=3.5, label=primary_label, color="tab:blue")
    if overlay_series is not None:
        ax.plot(
            overlay_series["s_grid"],
            overlay_series["psi_arr"],
            marker="s",
            markersize=3.0,
            label=overlay_label,
            color="tab:red",
        )
    for i, k0 in main_series["wp_start_state"].items():
        if 0 <= k0 < len(main_series["psi_arr"]):
            ax.annotate(
                f"WP{i}",
                (main_series["s_grid"][k0], main_series["psi_arr"][k0]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
            )
    ax.grid(True)
    if overlay_series is not None:
        ax.legend()
    ax.set_xlabel("s [m]")
    ax.set_ylabel("psi [rad]")
    ax.set_title("Heading vs s")

    ax = axes[1, 1]
    ax.step(main_series["s_cmd"], main_series["ds_arr"], where="post", label=primary_label, color="tab:blue")
    if overlay_series is not None:
        ax.step(
            overlay_series["s_cmd"],
            overlay_series["ds_arr"],
            where="post",
            label=overlay_label,
            color="tab:red",
        )
    for i, k0 in main_series["wp_start_step"].items():
        if 0 <= k0 < len(main_series["ds_arr"]):
            ax.annotate(
                f"WP{i}",
                (main_series["s_cmd"][k0], main_series["ds_arr"][k0]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
            )
    ax.grid(True)
    if overlay_series is not None:
        ax.legend()
    ax.set_xlabel("s [m]")
    ax.set_ylabel("ds [m]")
    ax.set_title("Step length profile")

    if args.save:
        fig.savefig(args.save, dpi=180, bbox_inches="tight")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
