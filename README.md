# RotaOptimalds

This repository contains two implementations of the same MPC-based route optimization workflow:

- `src/` and the CMake build: C++ / CasADi implementation
- `RotaOptimaldsPy/`: Python implementation with both CasADi and acados solver backends

Both versions use the same overall idea:

- receding-horizon nonlinear MPC
- waypoint tracking with heading and curvature targets
- clothoid-like planar propagation
- optional circular-obstacle detour waypoint generation
- CSV logging and plotting

## Repository Layout

- `src/`: C++ solver source
- `scenarios/`: C++ scenario files
- `docs/`: C++ example outputs
- `RotaOptimaldsPy/`: Python port
- `RotaOptimaldsPy/scenarios/`: Python scenario files
- `RotaOptimaldsPy/docs/`: Python example plots

## C++ Version

Build from the repository root:

```bash
cmake -S . -B build
cmake --build build -j
```

Run a C++ scenario:

```bash
./build/rota_optimal_ds --scenario scenarios/rotaoptimalds_default.ini
```

Plot a C++ run with the plotting script:

```bash
python3 RotaOptimaldsPy/plot_receding.py \
  --log receding_log.csv \
  --wp waypoints.csv \
  --scenario scenarios/rotaoptimalds_default.ini
```

## Python Version

The Python-specific guide is here:

[`RotaOptimaldsPy/README.md`](RotaOptimaldsPy/README.md)

Quick start:

```bash
python3 RotaOptimaldsPy/main.py --scenario RotaOptimaldsPy/scenarios/rotaoptimalds_default_alt.ini
```

The Python port supports:

- `--solver casadi`: Python / CasADi backend
- `--solver acados`: Python / acados backend

Example with acados:

```bash
python3 RotaOptimaldsPy/main.py --solver acados --scenario RotaOptimaldsPy/scenarios/rotaoptimalds_obstacle_alt.ini
```

## Notes

- The Python and C++ versions are algorithmically aligned, but small numerical differences may still appear because of runtime and solver details.
- Within the Python port, CasADi and acados backends solve the same MPC structure, but closed-loop trajectories can still differ slightly because of solver internals and termination behavior.
- `ds` is an optimization variable in both versions. It is not explicitly rewarded for staying large, so it can shrink near demanding turns, obstacle detours, or terminal conditions.
- Obstacle avoidance is handled by temporary detour waypoints around circular obstacles rather than hard collision constraints inside the MPC problem.
