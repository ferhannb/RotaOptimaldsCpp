from __future__ import annotations

import argparse
import csv
import sys

from rota_optimal_ds import MPCNumericClothoidCost
from rota_optimal_ds_acados import MPCAcadosClothoidCost
from scenario_parser import load_scenario_ini, make_default_scenario


def build_arg_parser():
    p = argparse.ArgumentParser(description="Python version of RotaOptimalds.")
    p.add_argument("--scenario", default=None, help="Load scenario ini file")
    p.add_argument("--out-log", default="receding_log.csv", help="Output receding log csv")
    p.add_argument("--out-wp", default="waypoints.csv", help="Output waypoint csv")
    p.add_argument("--solver", choices=["casadi", "acados"], default="casadi")
    return p


def build_solver_backend(solver_name: str, cfg):
    if solver_name == "casadi":
        return MPCNumericClothoidCost(cfg)
    if solver_name == "acados":
        return MPCAcadosClothoidCost(cfg)
    raise ValueError(f"Unknown solver backend: {solver_name}")


def write_waypoints_csv(waypoints, out_path: str):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "X", "Y"])
        for i, wp in enumerate(waypoints):
            writer.writerow([i, wp.X, wp.Y])


def main():
    args = build_arg_parser().parse_args()

    try:
        scenario = make_default_scenario() if not args.scenario else load_scenario_ini(args.scenario)

        mpc = build_solver_backend(args.solver, scenario.cfg)

        log = mpc.run_receding_horizon_multi(
            scenario.waypoints,
            scenario.initial_state,
            scenario.opts,
        )

        mpc.write_log_csv(log, args.out_log)
        write_waypoints_csv(scenario.waypoints, args.out_wp)

        last_xy = log.traj[-1]
        print(f"Solver backend: {args.solver}")
        print(f"Terminal [x,y,psi,K] = [{last_xy[0]}, {last_xy[1]}, {log.psi[-1]}, {log.K[-1]}]")
        print(f"Scenario source: {scenario.source}")
        print(f"Saved logs: {args.out_log}, {args.out_wp}")
        print(f"Active WP index: {log.active_wp}")
        print(f"Average solve time [s]: {log.mean_solve_time_s}")

        return 0

    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())