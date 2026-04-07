#pragma once

#include "rota_optimal_ds.hpp"

#include <string>
#include <vector>

struct ScenarioSpec {
  MPCConfig cfg;
  State4 initial_state;
  RecedingOptions opts;
  std::vector<Waypoint> waypoints;
  std::string source;
};

ScenarioSpec make_default_scenario();
ScenarioSpec load_scenario_ini(const std::string& path);

