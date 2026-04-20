# Contributing to RotaOptimalds

RotaOptimalds is currently best treated as a research and development codebase. Contributions are welcome when they improve reproducibility, solver behavior, scenario coverage, documentation quality, or usability for new users.

## Good Contribution Areas

- documentation improvements and clearer setup instructions
- scenario additions for route optimization or COLREG cases
- bug fixes in parsing, plotting, logging, or solver integration
- numerical robustness improvements with a clear explanation of the tradeoff
- benchmark and reproducibility improvements for the Python `casadi` and `acados` paths

## Before You Start

- keep changes focused and easy to review
- avoid unrelated formatting-only edits in touched files
- do not commit generated build folders or temporary CSV outputs
- include the scenario or command you used when reporting solver behavior

## Local Setup

### Python Path

```bash
cd RotaOptimaldsPy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

### C++ Path

You need a C++17 compiler, CMake 3.16+, and a local CasADi C++ installation.

```bash
cmake -S . -B build
cmake --build build -j
./build/rota_optimal_ds --scenario scenarios/rotaoptimalds_default.ini
```

If CasADi is installed in a custom location:

```bash
cmake -S . -B build -DCASADI_ROOT=/path/to/casadi
cmake --build build -j
```

## Submitting Changes

1. Create a focused branch for one logical change.
2. Update code, docs, plots, or scenarios only where needed.
3. Run the smallest relevant validation command for your change.
4. Summarize what changed, why it changed, and how you verified it.

## Pull Request Checklist

- the change has a clear purpose
- commands in docs were checked locally when practical
- generated artifacts are excluded unless they are intentional example outputs
- new scenarios or solver behavior changes are described in the PR
- plots or GIFs are updated only when they add user-facing value

## Reporting Bugs

Include as much of the following as possible:

- operating system
- compiler and CMake version for C++
- Python version and selected solver backend for Python
- exact command used
- scenario file used
- error output or unexpected behavior
- whether the issue is reproducible on the default scenarios

## Style Expectations

- prefer small, direct changes over broad refactors
- preserve existing scenario semantics unless the change explicitly updates them
- keep documentation practical and command-oriented
- explain numerical or modeling changes in concrete terms
