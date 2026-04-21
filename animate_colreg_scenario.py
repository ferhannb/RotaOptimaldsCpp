#!/usr/bin/env python3
"""Enhanced COLREG scenario animator.

Improvements over the original:
  - Ship-shaped polygon markers oriented to heading
  - Fading gradient trail (older positions fade out)
  - Separate DCPA and TCPA subplots with live tracking dots
  - Frame interpolation for smooth motion (--interp)
  - Heading arrows on each ship (FancyArrowPatch)
  - Professional colour scheme and layout
  - Risk-active background tinting on all panels
  - MP4 export support (requires ffmpeg)

Coordinate convention (matches plot_colreg_scenario.py):
  course_deg is a mathematical angle – 0 = East (+x), 90 = North (+y), CCW positive.
"""

import argparse
import csv
import math
import os
import subprocess
import tempfile

import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

# ── colour palette ────────────────────────────────────────────────────────────
OWN_COLOR    = "#1a6faf"
TARGET_COLOR = "#c0392b"
CPA_COLOR    = "#27ae60"
TCPA_COLOR   = "#8e44ad"
RISK_COLOR   = "#e67e22"
BG_SAFE      = "#f0f6fb"
BG_RISK      = "#fff4e0"
FIG_BG       = "#f4f8fb"
FIG_BG_RISK  = "#fff8f0"
AX_BG_SAFE   = "#ddeef8"
AX_BG_RISK   = "#f8e8d8"

# ── ship polygon (local frame: y+ = bow, x+ = starboard) ─────────────────────
_SHIP_LOCAL = np.array([
    [ 0.00,  1.00],   # bow
    [-0.38, -0.55],   # stern-port
    [ 0.00, -0.28],   # stern notch
    [ 0.38, -0.55],   # stern-starboard
], dtype=float)


# ── binary / scan helpers ─────────────────────────────────────────────────────

def resolve_binary(explicit_bin):
    if explicit_bin:
        return explicit_bin
    for candidate in ("build_colreg_check/rota_optimal_ds", "build/rota_optimal_ds"):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("rota_optimal_ds binary not found; build the project or pass --bin")


def run_colreg_scan(binary_path, scenario_path, dt, steps, out_log):
    cmd = [
        binary_path, "--scenario", scenario_path, "--colreg-scan",
        "--scan-dt", str(dt), "--scan-steps", str(steps),
        "--out-colreg-log", out_log,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def load_scan_csv(path, target_id=None):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)

    if not rows:
        raise ValueError("empty COLREG scan log")

    targets = sorted({r["target_id"] for r in rows})
    if target_id is None:
        if len(targets) > 1:
            raise ValueError(f"multiple targets present {targets}; pass --target")
        target_id = targets[0]

    rows = [r for r in rows if r["target_id"] == target_id]
    if not rows:
        raise ValueError(f"target_id {target_id} not found in COLREG scan log")

    return {
        "target_id": target_id,
        "step": np.array([int(r["step"]) for r in rows], dtype=int),
        "time_s": np.array([float(r["time_s"]) for r in rows]),
        "own_x": np.array([float(r["own_x"]) for r in rows]),
        "own_y": np.array([float(r["own_y"]) for r in rows]),
        "own_course_deg": np.array([float(r["own_course_deg"]) for r in rows]),
        "target_x": np.array([float(r["target_x"]) for r in rows]),
        "target_y": np.array([float(r["target_y"]) for r in rows]),
        "target_course_deg": np.array([float(r["target_course_deg"]) for r in rows]),
        "type": [r["type"] for r in rows],
        "role": [r["role"] for r in rows],
        "geometry_type": [r.get("geometry_type", r["type"]) for r in rows],
        "geometry_role": [r.get("geometry_role", r["role"]) for r in rows],
        "risk": np.array([int(r["risk"]) for r in rows], dtype=int),
        "dcpa": np.array([float(r["dcpa"]) for r in rows]),
        "tcpa": np.array([float(r["tcpa"]) for r in rows]),
        "alpha0_deg": np.array([float(r["alpha0_deg"]) for r in rows]),
        "beta0_deg": np.array([float(r["beta0_deg"]) for r in rows]),
    }


# ── data helpers ──────────────────────────────────────────────────────────────

def trim_to_approach_phase(data):
    ranges = np.hypot(data["target_x"] - data["own_x"], data["target_y"] - data["own_y"])
    if not ranges.size:
        return data
    last = int(np.argmin(ranges))
    return {
        k: (v[: last + 1] if isinstance(v, (np.ndarray, list)) else v)
        for k, v in data.items()
    }


def interpolate_data(data, factor):
    """Up-sample all channels by *factor* for smoother animation."""
    if factor <= 1:
        return data
    n = len(data["time_s"])
    t_old = np.arange(n, dtype=float)
    t_new = np.linspace(0, n - 1, (n - 1) * factor + 1)
    idx_nn = np.round(t_new).astype(int).clip(0, n - 1)
    out = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.floating):
            out[k] = np.interp(t_new, t_old, v)
        elif isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.integer):
            out[k] = v[idx_nn]
        elif isinstance(v, list):
            out[k] = [v[i] for i in idx_nn]
        else:
            out[k] = v
    return out


def compute_limits(data):
    xs = np.concatenate([data["own_x"], data["target_x"]])
    ys = np.concatenate([data["own_y"], data["target_y"]])
    rng = max(xs.max() - xs.min(), ys.max() - ys.min(), 1.0)
    pad = max(3.0, 0.16 * rng)
    return xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad


# ── drawing helpers ───────────────────────────────────────────────────────────

def _ship_verts(cx, cy, course_deg, scale):
    """Polygon vertices for a ship at (cx,cy) oriented to *course_deg*.

    course_deg follows mathematical convention (0=East/+x, 90=North/+y, CCW+).
    The ship's bow (local y+) is placed in the course direction.
    """
    # Rotation angle that maps local y+ (bow) onto (cos c, sin c) world direction.
    # Derived: theta = course_rad - pi/2  satisfies R(theta)*(0,1) = (cos c, sin c).
    theta = math.radians(course_deg - 90.0)
    c, s = math.cos(theta), math.sin(theta)
    pts = _SHIP_LOCAL * scale
    # Apply CCW rotation as right-multiply of row vectors: M = [[c,s],[-s,c]]
    rotated = pts @ np.array([[c, s], [-s, c]])
    return rotated + np.array([cx, cy])


def _heading_tip(cx, cy, course_deg, length):
    """World-space tip of a heading arrow of *length* starting at (cx,cy)."""
    cr = math.radians(course_deg)
    return cx + length * math.cos(cr), cy + length * math.sin(cr)


def _make_fading_trail(ax, color):
    lc = LineCollection([], linewidths=2.0, zorder=3)
    ax.add_collection(lc)
    return lc


def _update_fading_trail(lc, xs, ys, n_tail, color):
    tail_x = xs[-n_tail:] if len(xs) > n_tail else xs
    tail_y = ys[-n_tail:] if len(ys) > n_tail else ys
    if len(tail_x) < 2:
        lc.set_segments([])
        return
    pts = np.column_stack([tail_x, tail_y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    n = len(segs)
    base = mcolors.to_rgb(color)
    alphas = np.linspace(0.06, 0.88, n)
    lc.set_segments(segs)
    lc.set_colors([(*base, a) for a in alphas])


# ── main animation builder ────────────────────────────────────────────────────

def build_animation(data, title, interval_ms, trail_len, ship_scale):
    n = len(data["time_s"])
    x_min, x_max, y_min, y_max = compute_limits(data)

    if ship_scale <= 0:
        ship_scale = max(x_max - x_min, y_max - y_min) * 0.042

    hdg_len = max(x_max - x_min, y_max - y_min) * 0.09   # heading-arrow length

    fig = plt.figure(figsize=(14.5, 6.5), facecolor=FIG_BG)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.55, 1],
        height_ratios=[1, 1],
        left=0.05, right=0.97, top=0.92, bottom=0.09,
        hspace=0.42, wspace=0.38,
    )
    ax_xy   = fig.add_subplot(gs[:, 0])
    ax_dcpa = fig.add_subplot(gs[0, 1])
    ax_tcpa = fig.add_subplot(gs[1, 1])

    # ── spatial panel ─────────────────────────────────────────────────────────
    ax_xy.set_xlim(x_min, x_max)
    ax_xy.set_ylim(y_min, y_max)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_facecolor(AX_BG_SAFE)
    ax_xy.grid(True, color="white", linewidth=0.8, alpha=0.65)
    ax_xy.set_xlabel("x [m]", fontsize=10)
    ax_xy.set_ylabel("y [m]", fontsize=10)

    # fading trails
    own_trail = _make_fading_trail(ax_xy, OWN_COLOR)
    tgt_trail = _make_fading_trail(ax_xy, TARGET_COLOR)

    # ship polygons
    own_poly = mpatches.Polygon(
        _ship_verts(data["own_x"][0], data["own_y"][0], data["own_course_deg"][0], ship_scale),
        closed=True, facecolor=OWN_COLOR, edgecolor="white", linewidth=1.3, zorder=5,
    )
    tgt_poly = mpatches.Polygon(
        _ship_verts(data["target_x"][0], data["target_y"][0], data["target_course_deg"][0], ship_scale),
        closed=True, facecolor=TARGET_COLOR, edgecolor="white", linewidth=1.3, zorder=5,
    )
    ax_xy.add_patch(own_poly)
    ax_xy.add_patch(tgt_poly)

    # heading arrows (FancyArrowPatch – supports set_positions())
    _arrow_style = dict(
        arrowstyle=mpatches.ArrowStyle.Simple(head_width=ship_scale * 0.9,
                                              head_length=ship_scale * 0.7,
                                              tail_width=ship_scale * 0.18),
        linewidth=0,
        zorder=8,
    )
    ox0, oy0 = data["own_x"][0], data["own_y"][0]
    tx0, ty0 = data["target_x"][0], data["target_y"][0]
    own_hdg_tip = _heading_tip(ox0, oy0, data["own_course_deg"][0], hdg_len)
    tgt_hdg_tip = _heading_tip(tx0, ty0, data["target_course_deg"][0], hdg_len)

    own_arrow = mpatches.FancyArrowPatch(
        posA=(ox0, oy0), posB=own_hdg_tip, color=OWN_COLOR, **_arrow_style,
    )
    tgt_arrow = mpatches.FancyArrowPatch(
        posA=(tx0, ty0), posB=tgt_hdg_tip, color=TARGET_COLOR, **_arrow_style,
    )
    ax_xy.add_patch(own_arrow)
    ax_xy.add_patch(tgt_arrow)

    # CPA range line
    cpa_line, = ax_xy.plot(
        [ox0, tx0], [oy0, ty0],
        color="#444444", linewidth=0.9, linestyle=":", alpha=0.65, zorder=4,
    )

    # ship labels
    lbl_off = ship_scale * 0.95
    own_lbl = ax_xy.text(
        ox0, oy0 + lbl_off, "own",
        fontsize=8, ha="center", va="bottom",
        color=OWN_COLOR, fontweight="bold", zorder=6,
    )
    tgt_lbl = ax_xy.text(
        tx0, ty0 + lbl_off, data["target_id"],
        fontsize=8, ha="center", va="bottom",
        color=TARGET_COLOR, fontweight="bold", zorder=6,
    )

    # status info box
    status = ax_xy.text(
        0.02, 0.98, "",
        transform=ax_xy.transAxes,
        va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", alpha=0.97),
        zorder=10,
        fontfamily="monospace",
    )

    ax_xy.legend(
        handles=[
            mpatches.Patch(color=OWN_COLOR, label="own ship"),
            mpatches.Patch(color=TARGET_COLOR, label=data["target_id"]),
        ],
        loc="lower right", fontsize=9,
    )

    # ── DCPA panel ────────────────────────────────────────────────────────────
    ax_dcpa.plot(data["time_s"], data["dcpa"], color=CPA_COLOR, linewidth=1.8, label="DCPA")
    ax_dcpa.fill_between(
        data["time_s"], 0, data["dcpa"],
        where=data["risk"] > 0, color=RISK_COLOR, alpha=0.22, label="risk",
    )
    dcpa_cur = ax_dcpa.axvline(data["time_s"][0], color="#333333", linestyle="--", linewidth=1.0)
    dcpa_dot, = ax_dcpa.plot(
        [data["time_s"][0]], [data["dcpa"][0]],
        "o", color=CPA_COLOR, markersize=6, zorder=5,
    )
    ax_dcpa.set_ylabel("DCPA [m]", fontsize=9)
    ax_dcpa.set_xlabel("time [s]", fontsize=9)
    ax_dcpa.grid(True, alpha=0.3)
    ax_dcpa.legend(fontsize=8, loc="best")
    ax_dcpa.set_title("DCPA over time", fontsize=9)
    ax_dcpa.set_facecolor("#f5faf5")

    # ── TCPA panel ────────────────────────────────────────────────────────────
    ax_tcpa.plot(data["time_s"], data["tcpa"], color=TCPA_COLOR, linewidth=1.8, label="TCPA")
    ax_tcpa.fill_between(
        data["time_s"], 0, data["tcpa"],
        where=data["risk"] > 0, color=RISK_COLOR, alpha=0.22, label="risk",
    )
    tcpa_cur = ax_tcpa.axvline(data["time_s"][0], color="#333333", linestyle="--", linewidth=1.0)
    tcpa_dot, = ax_tcpa.plot(
        [data["time_s"][0]], [data["tcpa"][0]],
        "o", color=TCPA_COLOR, markersize=6, zorder=5,
    )
    ax_tcpa.set_ylabel("TCPA [s]", fontsize=9)
    ax_tcpa.set_xlabel("time [s]", fontsize=9)
    ax_tcpa.grid(True, alpha=0.3)
    ax_tcpa.legend(fontsize=8, loc="best")
    ax_tcpa.set_title("TCPA over time", fontsize=9)
    ax_tcpa.set_facecolor("#f8f4fc")

    # ── update function ───────────────────────────────────────────────────────
    def update(i):
        ox, oy = data["own_x"][i], data["own_y"][i]
        tx, ty = data["target_x"][i], data["target_y"][i]
        oc = data["own_course_deg"][i]
        tc = data["target_course_deg"][i]
        t  = data["time_s"][i]
        risk = bool(data["risk"][i])

        # fading trails
        _update_fading_trail(own_trail, data["own_x"][: i + 1], data["own_y"][: i + 1], trail_len, OWN_COLOR)
        _update_fading_trail(tgt_trail, data["target_x"][: i + 1], data["target_y"][: i + 1], trail_len, TARGET_COLOR)

        # ship polygons
        own_poly.set_xy(_ship_verts(ox, oy, oc, ship_scale))
        tgt_poly.set_xy(_ship_verts(tx, ty, tc, ship_scale))

        # heading arrows
        own_arrow.set_positions((ox, oy), _heading_tip(ox, oy, oc, hdg_len))
        tgt_arrow.set_positions((tx, ty), _heading_tip(tx, ty, tc, hdg_len))

        # CPA connecting line
        cpa_line.set_data([ox, tx], [oy, ty])

        # ship labels
        own_lbl.set_position((ox, oy + lbl_off))
        tgt_lbl.set_position((tx, ty + lbl_off))

        # status box
        status.set_text(
            f"t      = {t:6.2f} s\n"
            f"type   = {data['type'][i]}\n"
            f"role   = {data['role'][i]}\n"
            f"geom   = {data['geometry_type'][i]}\n"
            f"risk   = {'YES' if risk else 'no'}\n"
            f"DCPA   = {data['dcpa'][i]:.2f} m\n"
            f"TCPA   = {data['tcpa'][i]:.2f} s\n"
            f"\u03b1\u2080     = {data['alpha0_deg'][i]:.1f}\u00b0\n"
            f"\u03b2\u2080     = {data['beta0_deg'][i]:.1f}\u00b0"
        )
        status.set_bbox(dict(
            boxstyle="round,pad=0.4",
            fc=BG_RISK if risk else BG_SAFE,
            ec=RISK_COLOR if risk else "0.6",
            alpha=0.97,
        ))

        # background tinting
        fig.set_facecolor(FIG_BG_RISK if risk else FIG_BG)
        ax_xy.set_facecolor(AX_BG_RISK if risk else AX_BG_SAFE)

        # metric panel cursors and live dots
        dcpa_cur.set_xdata([t, t])
        dcpa_dot.set_data([t], [data["dcpa"][i]])
        tcpa_cur.set_xdata([t, t])
        tcpa_dot.set_data([t], [data["tcpa"][i]])

        return (
            own_trail, tgt_trail,
            own_poly, tgt_poly,
            own_arrow, tgt_arrow,
            cpa_line, own_lbl, tgt_lbl, status,
            dcpa_cur, dcpa_dot, tcpa_cur, tcpa_dot,
        )

    ani = animation.FuncAnimation(
        fig, update, frames=n,
        interval=interval_ms, blit=False, repeat=True,
    )
    return fig, ani


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Enhanced COLREG scenario animator")
    parser.add_argument("--scenario", required=True, help="scenario .ini path")
    parser.add_argument("--bin", default=None, help="path to rota_optimal_ds binary")
    parser.add_argument("--dt", type=float, default=0.5, help="scan time step [s]")
    parser.add_argument("--steps", type=int, default=80, help="number of scan steps")
    parser.add_argument("--target", default=None, help="target_id when log has multiple targets")
    parser.add_argument("--log", default=None, help="existing COLREG scan csv; skips C++ run")
    parser.add_argument("--save", default=None, help="output file: .gif / .mp4 / .html")
    parser.add_argument("--fps", type=int, default=15, help="frame rate when saving (default 15)")
    parser.add_argument(
        "--interval-ms", type=int, default=80,
        help="frame interval in ms for interactive display (default 80)",
    )
    parser.add_argument(
        "--interp", type=int, default=4,
        help="interpolation factor for smoother motion (default 4; use 1 to disable)",
    )
    parser.add_argument(
        "--trail-len", type=int, default=40,
        help="fading trail length in frames (default 40)",
    )
    parser.add_argument(
        "--ship-scale", type=float, default=-1,
        help="ship polygon scale factor; <=0 = auto (default)",
    )
    parser.add_argument(
        "--full-window", action="store_true",
        help="include post-CPA separation in animation",
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
    if args.interp > 1:
        data = interpolate_data(data, args.interp)

    fig, ani = build_animation(
        data,
        title=f"COLREG Animation \u2014 {os.path.basename(args.scenario)}",
        interval_ms=args.interval_ms,
        trail_len=args.trail_len,
        ship_scale=args.ship_scale,
    )

    if args.save:
        ext = os.path.splitext(args.save)[1].lower()
        if ext == ".gif":
            ani.save(args.save, writer=animation.PillowWriter(fps=args.fps))
        elif ext == ".mp4":
            ani.save(args.save, writer=animation.FFMpegWriter(fps=args.fps, bitrate=1800))
        elif ext == ".html":
            ani.save(args.save, writer="html")
        else:
            raise ValueError("unsupported save format; use .gif, .mp4 or .html")
        print(f"saved \u2192 {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
