from __future__ import annotations

import argparse
import csv
import sys

from rota_optimal_ds import MPCNumericClothoidCost
from scenario_parser import load_scenario_ini, make_default_scenario


def build_arg_parser():
    p = argparse.ArgumentParser(description="Python/CasADi version of RotaOptimaldsCpp.")
    p.add_argument("--scenario", default=None, help="Load scenario ini file")
    p.add_argument("--out-log", default="receding_log.csv", help="Output receding log csv")
    p.add_argument("--out-wp", default="waypoints.csv", help="Output waypoint csv")
    return p


def main():
    args = build_arg_parser().parse_args()
    try:
        scenario = make_default_scenario() if not args.scenario else load_scenario_ini(args.scenario)

        mpc = MPCNumericClothoidCost(scenario.cfg)
        log = mpc.run_receding_horizon_multi(scenario.waypoints, scenario.initial_state, scenario.opts)
        mpc.write_log_csv(log, args.out_log)

        with open(args.out_wp, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["idx", "X", "Y"])
            for i, wp in enumerate(scenario.waypoints):
                writer.writerow([i, wp.X, wp.Y])

        last_xy = log.traj[-1]
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
