# RotaOptimaldsPy

`RotaOptimaldsPy` is the Python/CasADi port of the C++ MPC workflow in this repository. It keeps the same overall structure:

- receding-horizon nonlinear MPC
- waypoint tracking with heading and curvature targets
- optional circular-obstacle detour waypoint generation
- CSV logging and plotting

The goal of this folder is to make the solver easier to inspect, test, and compare against the C++ implementation.

## Files

- `main.py`: CLI entry point
- `rota_optimal_ds.py`: MPC model, dynamics, warm start, receding-horizon loop
- `scenario_parser.py`: `.ini` scenario parser
- `obstacle_avoidance.py`: circular obstacle detour waypoint logic
- `plot_receding.py`: plot trajectory, curvature, heading, and `ds`
- `scenarios/`: sample scenario files

## Requirements

Install dependencies in a Python environment that has `casadi`, `numpy`, and `matplotlib`.

Example:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r RotaOptimaldsPy/requirements.txt
```

If you already have a working virtual environment, use that interpreter directly.

## Running

From the repository root:

```bash
python3 RotaOptimaldsPy/main.py --scenario RotaOptimaldsPy/scenarios/rotaoptimalds_default_alt.ini
```

This writes:

- `receding_log.csv`
- `waypoints.csv`

## Plotting

```bash
python3 RotaOptimaldsPy/plot_receding.py \
  --log receding_log.csv \
  --wp waypoints.csv \
  --scenario RotaOptimaldsPy/scenarios/rotaoptimalds_default_alt.ini
```

To save the plot without opening a window:

```bash
python3 RotaOptimaldsPy/plot_receding.py \
  --log receding_log.csv \
  --wp waypoints.csv \
  --scenario RotaOptimaldsPy/scenarios/rotaoptimalds_default_alt.ini \
  --save rota_plot.png \
  --no-show
```

## Sample Scenarios

- `rotaoptimalds_default_alt.ini`: no-obstacle waypoint tracking example
- `rotaoptimalds_obstacle.ini`: basic obstacle-avoidance example
- `rotaoptimalds_obstacle_alt.ini`: denser obstacle-avoidance example
- `rotaoptimalds_samepoint_reverse_heading.ini`: same-position reverse-heading case

## Notes

- The Python and C++ versions are algorithmically aligned, but small numerical differences can still appear because of runtime and solver details.
- `ds` is an optimization variable; it is not forced to stay large. It often shrinks near demanding turns or near terminal conditions because that reduces cost.
- Obstacle avoidance here is not a hard MPC collision constraint. The solver uses temporary detour waypoints around circular obstacles.
