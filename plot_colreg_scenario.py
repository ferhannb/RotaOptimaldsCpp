#!/usr/bin/env python3
import argparse
import math
import os
import re
import subprocess

import matplotlib.pyplot as plt
import numpy as np


def parse_bool(text):
    value = text.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"invalid bool value: {text}")


def parse_ship_line(value):
    toks = [tok.strip() for tok in value.split(",")]
    if len(toks) < 5:
        raise ValueError("ship entry requires name,x,y,course_deg,speed[,length,beam]")
    return {
        "id": toks[0],
        "x": float(toks[1]),
        "y": float(toks[2]),
        "course_deg": float(toks[3]),
        "speed": float(toks[4]),
        "length": float(toks[5]) if len(toks) > 5 and toks[5] else 0.0,
        "beam": float(toks[6]) if len(toks) > 6 and toks[6] else 0.0,
    }


def load_colreg_scenario(path):
    scenario = {
        "colreg_only": False,
        "colreg_max_tcpa": 20.0,
        "own_ship": None,
        "target_ships": [],
    }
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, value = [part.strip() for part in line.split("=", 1)]
            key = key.lower()
            if key == "colreg_only":
                scenario["colreg_only"] = parse_bool(value)
            elif key == "colreg_max_tcpa":
                scenario["colreg_max_tcpa"] = float(value)
            elif key == "own_ship":
                scenario["own_ship"] = parse_ship_line(value)
            elif key == "target_ship":
                scenario["target_ships"].append(parse_ship_line(value))

    if scenario["own_ship"] is None:
        raise ValueError("scenario must define own_ship")
    if not scenario["target_ships"]:
        raise ValueError("scenario must define at least one target_ship")
    return scenario


def resolve_binary(explicit_bin):
    if explicit_bin:
        return explicit_bin

    candidates = [
        "build_colreg_check/rota_optimal_ds",
        "build/rota_optimal_ds",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("rota_optimal_ds binary not found; build the project or pass --bin")


def run_classifier(binary_path, scenario_path):
    cmd = [binary_path, "--scenario", scenario_path]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    reports = {}
    current_target = None
    for line in result.stdout.splitlines():
        if not line.startswith("COLREG "):
            continue
        match = re.search(
            r"target=(?P<target>\S+)\s+type=(?P<type>\S+)\s+role=(?P<role>\S+)\s+"
            r"risk=(?P<risk>\S+)\s+DCPA=(?P<dcpa>\S+)\s+TCPA=(?P<tcpa>\S+)\s+"
            r"alpha0_deg=(?P<alpha>\S+)\s+beta0_deg=(?P<beta>\S+)",
            line,
        )
        if match:
            current_target = match.group("target")
            reports[current_target] = {
                "type": match.group("type"),
                "role": match.group("role"),
                "risk": match.group("risk"),
                "dcpa": float(match.group("dcpa")),
                "tcpa": float(match.group("tcpa")),
                "alpha0_deg": float(match.group("alpha")),
                "beta0_deg": float(match.group("beta")),
                "rationale": "",
            }
            continue
        if line.startswith("  rationale: ") and current_target in reports:
            reports[current_target]["rationale"] = line.split(":", 1)[1].strip()
    return reports


def velocity_components(ship):
    course = math.radians(ship["course_deg"])
    return ship["speed"] * math.cos(course), ship["speed"] * math.sin(course)


def cpa_between(own_ship, target_ship):
    rx = target_ship["x"] - own_ship["x"]
    ry = target_ship["y"] - own_ship["y"]
    vox, voy = velocity_components(own_ship)
    vtx, vty = velocity_components(target_ship)
    rvx = vtx - vox
    rvy = vty - voy
    rv2 = rvx * rvx + rvy * rvy

    if rv2 < 1e-12:
        tcpa = 0.0
    else:
        tcpa = -((rx * rvx) + (ry * rvy)) / rv2

    own_cpa = (own_ship["x"] + vox * tcpa, own_ship["y"] + voy * tcpa)
    target_cpa = (target_ship["x"] + vtx * tcpa, target_ship["y"] + vty * tcpa)
    dcpa = math.hypot(target_cpa[0] - own_cpa[0], target_cpa[1] - own_cpa[1])
    return tcpa, dcpa, own_cpa, target_cpa


def plot_ship(ax, ship, color, label):
    ax.scatter([ship["x"]], [ship["y"]], c=color, s=80, label=label, zorder=4)
    course = math.radians(ship["course_deg"])
    arrow_len = max(1.5, ship["speed"])
    ax.arrow(
        ship["x"],
        ship["y"],
        arrow_len * math.cos(course),
        arrow_len * math.sin(course),
        color=color,
        width=0.06,
        head_width=0.45,
        length_includes_head=True,
        zorder=3,
    )


def plot_target_panel(ax, own_ship, target_ship, report, horizon):
    own_vx, own_vy = velocity_components(own_ship)
    target_vx, target_vy = velocity_components(target_ship)

    t = np.linspace(0.0, horizon, 100)
    own_x = own_ship["x"] + own_vx * t
    own_y = own_ship["y"] + own_vy * t
    target_x = target_ship["x"] + target_vx * t
    target_y = target_ship["y"] + target_vy * t

    ax.plot(own_x, own_y, color="tab:blue", linestyle="--", alpha=0.8)
    ax.plot(target_x, target_y, color="tab:red", linestyle="--", alpha=0.8)
    plot_ship(ax, own_ship, "tab:blue", "own ship")
    plot_ship(ax, target_ship, "tab:red", target_ship["id"])
    ax.plot(
        [own_ship["x"], target_ship["x"]],
        [own_ship["y"], target_ship["y"]],
        color="0.4",
        linestyle=":",
        linewidth=1.2,
    )

    tcpa, dcpa_calc, own_cpa, target_cpa = cpa_between(own_ship, target_ship)
    if tcpa >= 0.0:
        ax.scatter([own_cpa[0]], [own_cpa[1]], marker="x", c="tab:blue", s=60, zorder=5)
        ax.scatter([target_cpa[0]], [target_cpa[1]], marker="x", c="tab:red", s=60, zorder=5)
        ax.plot(
            [own_cpa[0], target_cpa[0]],
            [own_cpa[1], target_cpa[1]],
            color="black",
            linewidth=1.2,
            alpha=0.7,
        )

    title = target_ship["id"]
    if report:
        title += f" | {report['type']} | {report['role']}"
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    info_lines = [
        f"own course={own_ship['course_deg']:.1f} deg speed={own_ship['speed']:.1f}",
        f"target course={target_ship['course_deg']:.1f} deg speed={target_ship['speed']:.1f}",
        f"TCPA={report['tcpa']:.2f}" if report else f"TCPA={tcpa:.2f}",
        f"DCPA={report['dcpa']:.2f}" if report else f"DCPA={dcpa_calc:.2f}",
    ]
    if report:
        info_lines.append(f"alpha0={report['alpha0_deg']:.1f} deg")
        info_lines.append(f"beta0={report['beta0_deg']:.1f} deg")
        info_lines.append(f"risk={report['risk']}")
    ax.text(
        0.02,
        0.98,
        "\n".join(info_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.95),
    )
    if report and report["rationale"]:
        ax.text(
            0.02,
            0.02,
            report["rationale"],
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff8e1", ec="0.7", alpha=0.95),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, help="COLREG scenario ini path")
    parser.add_argument("--bin", default=None, help="path to rota_optimal_ds binary")
    parser.add_argument("--save", default=None, help="save figure to image file")
    parser.add_argument("--no-show", action="store_true", help="do not open interactive window")
    args = parser.parse_args()

    scenario = load_colreg_scenario(args.scenario)
    binary_path = resolve_binary(args.bin)
    reports = run_classifier(binary_path, args.scenario)

    n_targets = len(scenario["target_ships"])
    fig, axes = plt.subplots(
        n_targets,
        1,
        figsize=(11, max(5, 4.5 * n_targets)),
        constrained_layout=True,
    )
    if n_targets == 1:
        axes = [axes]

    horizon = max(5.0, scenario["colreg_max_tcpa"] * 1.2)
    for ax, target_ship in zip(axes, scenario["target_ships"]):
        report = reports.get(target_ship["id"])
        plot_target_panel(ax, scenario["own_ship"], target_ship, report, horizon)

    fig.suptitle(
        f"COLREG Scenario View: {os.path.basename(args.scenario)}",
        fontsize=14,
    )

    if args.save:
        fig.savefig(args.save, dpi=180, bbox_inches="tight")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
