#pragma once

#include "colreg_types.hpp"

struct ColregClassifierConfig {
  double risk_dcpa = 0.5;
  double max_tcpa = 20.0;
  double alpha_crit_13_deg = 45.0;
  double alpha_crit_14_deg = 13.0;
  double alpha_crit_15_deg = 10.0;
  double overtaking_sector_deg = 112.5;
  double crossing_sector_deg = 112.5;
};

RelativePose compute_initial_pose(const OwnShip& own_ship, const TargetShip& target_ship);
CPAInfo compute_cpa(const OwnShip& own_ship, const TargetShip& target_ship);
ColregEncounter classify_encounter(
    const OwnShip& own_ship,
    const TargetShip& target_ship,
    const ColregClassifierConfig& cfg = ColregClassifierConfig{});
