#pragma once

#include "colreg_classifier.hpp"
#include "rota_optimal_ds.hpp"

#include <optional>
#include <string>
#include <vector>

struct ScenarioSpec {
  MPCConfig cfg;
  State4 initial_state;
  RecedingOptions opts;
  std::vector<Waypoint> waypoints;
  std::optional<OwnShip> own_ship;
  std::vector<TargetShip> target_ships;
  ColregClassifierConfig colreg_cfg;
  bool colreg_only = false;
  std::string special_maneuver = "none";
  std::string turn_dir = "auto";
  std::optional<double> maneuver_radius;
  std::string source;
};

ScenarioSpec make_default_scenario();
ScenarioSpec load_scenario_ini(const std::string& path);
