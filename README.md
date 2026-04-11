# RotaOptimalds

This repository contains two implementations of the same MPC-based route optimization workflow:

- `src/` and the CMake build: C++ / CasADi implementation
- `RotaOptimaldsPy/`: Python implementation with both `CasADi` and `acados` backends

Both versions use the same overall idea:

- receding-horizon nonlinear MPC
- waypoint tracking with heading and curvature targets
- clothoid-like planar propagation
- optional circular-obstacle detour waypoint generation
- CSV logging and plotting

## Modeling View

The controller is built around a curvilinear geometric representation of motion so that maneuverability limits appear directly inside the planning variables, instead of being pushed afterward to a low-level tracker.

In that view, the path is parameterized by arc length `s` and curvature `kappa(s)`, not by time alone. This makes vessel feasibility much easier to encode: minimum turning radius becomes a curvature bound, and speed affects timing through `ds` rather than changing the geometric meaning of the path itself. This repository adopts that logic in an MPC form that is practical for repeated closed-loop solving.

- The predicted state is carried with `x`, `y`, heading-like angle `psi`, and curvature `K`.
- The optimizer chooses `Kcmd` and the spatial increment `ds`, so path progression is itself part of the optimization.
- Curvature is not treated as a cosmetic smoothing variable; it is part of the internal model and directly determines how the vessel turns from one prediction stage to the next.
- The update from one stage to the next is computed with a clothoid-like, sinc-regularized spatial propagation so that nearly straight and curved motion are handled by one continuous formula.

Conceptually, the model starts from the curvilinear relations `dx/ds = cos(chi)`, `dy/ds = sin(chi)`, and `dchi/ds = kappa(s)`. In this repository, that idea is encapsulated in a control-oriented implementation where `psi` and `K` are propagated explicitly, and the next curvature is obtained through a slew-rate-limited command law. This is the practical bridge between geometric path description and implementable vessel motion prediction.

The cost function is built around that same philosophy:

- terminal and waypoint-related position errors are penalized so the optimizer still reaches the desired route in Cartesian space
- heading and terminal curvature targets are penalized so geometric arrival conditions remain meaningful
- curvature effort `K`, curvature command effort `Kcmd`, and command variation are regularized to avoid aggressive steering profiles
- `ds` smoothness is penalized so spatial progression does not oscillate from one stage to the next

The constraints are also curvilinear in spirit:

- curvature bounds encode turning-radius feasibility
- slew-rate limits on curvature changes encode steering-rate limitations
- lower and upper bounds on `ds`, together with optional `ds` jump limits, keep spatial advancement physically reasonable
- obstacle handling is implemented at the route level through temporary detour waypoints, then refined by the same curvature-aware MPC model

The result is that the optimizer does not merely fit a Cartesian curve through waypoints. It constructs a dynamically consistent motion primitive in which geometry, steering smoothness, and feasible vessel turning behavior are solved together inside the prediction model.

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
python3 RotaOptimaldsPy/main.py --solver casadi --scenario RotaOptimaldsPy/scenarios/rotaoptimalds_default_alt.ini
python3 RotaOptimaldsPy/main.py --solver acados --scenario RotaOptimaldsPy/scenarios/rotaoptimalds_obstacle_alt.ini
```

Example Python outputs:

- [Trajectory overlay](RotaOptimaldsPy/overlay_receding_plot.png)
- [Solver speed comparison](RotaOptimaldsPy/solver_speed_comparison.png)

## Notes

- The Python and C++ versions are algorithmically aligned, but small numerical differences may still appear because of runtime and solver details.
- The Python version supports both a `CasADi` backend and an `acados` backend for the same MPC structure.
- In Python, `acados` is expected to reduce solve time relative to `CasADi`, while still following the same scenario and controller structure.
- `ds` is an optimization variable in both versions. It is not explicitly rewarded for staying large, so it can shrink near demanding turns, obstacle detours, or terminal conditions.
- Obstacle avoidance is handled by temporary detour waypoints around circular obstacles rather than hard collision constraints inside the MPC problem.
