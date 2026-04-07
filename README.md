# RotaOptimaldsCpp

C++/CasADi implementation of the `Fresnel/RotaOptimalds.py` workflow.

This project solves receding-horizon route optimization with waypoint tracking and obstacle avoidance, and exports the results as CSV files for post-processing and plotting.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

You can specify the CasADi root directory with `CASADI_ROOT`:

```bash
cmake -S . -B build -DCASADI_ROOT=/path/to/casadi
```

## Run

```bash
./build/rota_optimal_ds
```

After execution, the program generates `receding_log.csv` and `waypoints.csv`.

To run with a scenario file from the CLI:

```bash
./build/rota_optimal_ds \
  --scenario scenarios/rotaoptimalds_default.ini \
  --out-log receding_log.csv \
  --out-wp waypoints.csv
```

Help:

```bash
./build/rota_optimal_ds --help
```

> Note: If the `ipopt` plugin is not available on your system, the code automatically falls back to `sqpmethod`.
> For behavior closer to the Python version, installing `ipopt` is recommended.

## Windows Setup

### 1. Requirements

- Visual Studio 2022 with Desktop development with C++
- CMake 3.16 or newer
- CasADi C++ distribution containing `include/`, `lib/`, or `bin/`

Recommended folder layout:

```text
RotaOptimaldsCpp/
  third_party/
    casadi/
      include/
      lib/
      bin/
```

### 2. Build (PowerShell)

```powershell
cd C:\path\to\RotaOptimaldsCpp
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCASADI_ROOT="$PWD\\third_party\\casadi"
cmake --build build --config Release
```

### 3. Run

```powershell
.\build\Release\rota_optimal_ds.exe --scenario .\scenarios\rotaoptimalds_default.ini
```

If `ROTA_COPY_CASADI_RUNTIME=ON` (default), CMake tries to copy CasADi-related DLLs next to the executable.

### 4. Single-Folder Distribution

```powershell
cmake --install build --config Release --prefix .\dist
```

You can zip the `dist` folder and move it to another Windows machine.
If the target machine is missing `vcruntime`, install the Visual C++ Redistributable.

## Plot

```bash
python3 plot_receding.py --log receding_log.csv --wp waypoints.csv --scenario scenarios/rotaoptimalds_obstacle.ini
```

## Scenario File

- Default example: `scenarios/rotaoptimalds_default.ini`
- `waypoint` line format:
  `waypoint = X,Y,psig,Kf,tol,use_Kf,w_wp,hit_scale`
- `obstacle` line format:
  `obstacle = cx,cy,radius[,enabled]`
- For multiple obstacles, repeat the `obstacle` line.
- For batch loading:
  `obstacles_csv = scenarios/obstacles_many.csv`
  CSV format: `cx,cy,radius[,enabled]` (header optional)
- Obstacle-avoidance settings:
  `enable_obstacle_avoidance`, `obstacle_clearance`, `obstacle_trigger_margin`, `obstacle_waypoint_tol`
- For empty or optional fields, use `none` or leave them blank.
- Example obstacle scenario: `scenarios/rotaoptimalds_obstacle.ini`

## Contents

- `src/rota_optimal_ds.hpp/.cpp`
  - MPC model (Opti + IPOPT)
  - Clothoid dynamics with fixed-ramp `K` update
  - Multi-waypoint receding-horizon loop
  - Warm-start and solution-shift logic
- `src/main.cpp`
  - CLI entry point and scenario-file based execution
- `src/scenario_parser.hpp/.cpp`
  - INI-style scenario parser
- `plot_receding.py`
  - Generates the Python 2x2 plot set from C++ output
