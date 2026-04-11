#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np

from rota_optimal_ds import MPCNumericClothoidCost
from rota_optimal_ds_acados import MPCAcadosClothoidCost
from scenario_parser import load_scenario_ini, make_default_scenario


def build_arg_parser():
    p = argparse.ArgumentParser(description="Compare CasADi and acados solve times on the same scenario.")
    p.add_argument("--scenario", default=None, help="Load scenario ini file")
    p.add_argument("--save", default="solver_speed_comparison.png", help="Output image path")
    p.add_argument("--no-show", action="store_true", help="Do not open an interactive plot window")
    p.add_argument(
        "--show-solver-output",
        action="store_true",
        help="Do not suppress verbose solver stdout during comparison",
    )
    return p


def run_solver(name, solver_cls, scenario, suppress_output):
    solver = solver_cls(scenario.cfg)
    if suppress_output:
        with redirect_stdout(io.StringIO()):
            log = solver.run_receding_horizon_multi(
                scenario.waypoints,
                scenario.initial_state,
                scenario.opts,
            )
    else:
        log = solver.run_receding_horizon_multi(
            scenario.waypoints,
            scenario.initial_state,
            scenario.opts,
        )

    last_xy = log.traj[-1]
    return {
        "name": name,
        "log": log,
        "terminal": {
            "x": float(last_xy[0]),
            "y": float(last_xy[1]),
            "psi": float(log.psi[-1]),
            "K": float(log.K[-1]),
        },
        "mean_solve_time_s": float(log.mean_solve_time_s),
        "solve_time_s": np.asarray(log.solve_time_s, dtype=float),
    }


def summarize_delta(casadi_result, acados_result):
    c = casadi_result["terminal"]
    a = acados_result["terminal"]
    speedup = np.inf
    if acados_result["mean_solve_time_s"] > 0.0:
        speedup = casadi_result["mean_solve_time_s"] / acados_result["mean_solve_time_s"]
    return {
        "dx": a["x"] - c["x"],
        "dy": a["y"] - c["y"],
        "dpsi": a["psi"] - c["psi"],
        "dK": a["K"] - c["K"],
        "speedup": float(speedup),
    }


def plot_results(scenario_source, casadi_result, acados_result, save_path=None, show_plot=True):
    casadi_ms = 1e3 * casadi_result["solve_time_s"]
    acados_ms = 1e3 * acados_result["solve_time_s"]
    delta = summarize_delta(casadi_result, acados_result)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    ax = axes[0]
    ax.plot(np.arange(1, len(casadi_ms) + 1), casadi_ms, label="CasADi", linewidth=2.0, color="#1f77b4")
    ax.plot(np.arange(1, len(acados_ms) + 1), acados_ms, label="acados", linewidth=2.0, color="#d62728")
    ax.set_xlabel("MPC step")
    ax.set_ylabel("Solve time [ms]")
    ax.set_title("Per-step Solve Time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    stats_lines = [
        f"Scenario: {scenario_source}",
        f"CasADi mean: {1e3 * casadi_result['mean_solve_time_s']:.2f} ms",
        f"acados mean: {1e3 * acados_result['mean_solve_time_s']:.2f} ms",
        f"Speedup: {delta['speedup']:.2f}x",
        f"Terminal delta: dx={delta['dx']:.4f}, dy={delta['dy']:.4f}, dpsi={delta['dpsi']:.4f}, dK={delta['dK']:.4f}",
    ]
    ax.text(
        0.01,
        0.99,
        "\n".join(stats_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95},
    )

    ax = axes[1]
    labels = ["CasADi", "acados"]
    mean_ms = [
        1e3 * casadi_result["mean_solve_time_s"],
        1e3 * acados_result["mean_solve_time_s"],
    ]
    bars = ax.bar(labels, mean_ms, color=["#1f77b4", "#d62728"], width=0.55)
    ax.set_ylabel("Mean solve time [ms]")
    ax.set_title("Average Solve Time")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, value in zip(bars, mean_ms):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    if np.isfinite(delta["speedup"]):
        ymax = max(mean_ms) if mean_ms else 1.0
        ax.text(
            0.5,
            ymax * 0.92,
            f"acados is {delta['speedup']:.2f}x faster on average",
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#fff3cd", "edgecolor": "#d4b106", "alpha": 0.95},
        )

    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    if show_plot:
        plt.show()


def main():
    args = build_arg_parser().parse_args()
    scenario = make_default_scenario() if not args.scenario else load_scenario_ini(args.scenario)
    suppress_output = not args.show_solver_output

    casadi_result = run_solver("casadi", MPCNumericClothoidCost, scenario, suppress_output)
    acados_result = run_solver("acados", MPCAcadosClothoidCost, scenario, suppress_output)

    delta = summarize_delta(casadi_result, acados_result)

    print(f"Scenario source: {scenario.source}")
    print(f"CasADi mean solve time [ms]: {1e3 * casadi_result['mean_solve_time_s']:.3f}")
    print(f"acados mean solve time [ms]: {1e3 * acados_result['mean_solve_time_s']:.3f}")
    print(f"Average speedup [CasADi/acados]: {delta['speedup']:.3f}x")
    print(
        "Terminal delta [acados - casadi] = "
        f"[dx={delta['dx']:.6f}, dy={delta['dy']:.6f}, dpsi={delta['dpsi']:.6f}, dK={delta['dK']:.6f}]"
    )

    plot_results(
        scenario.source,
        casadi_result,
        acados_result,
        save_path=args.save,
        show_plot=not args.no_show,
    )


if __name__ == "__main__":
    main()
