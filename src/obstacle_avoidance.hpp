#pragma once

#include "rota_optimal_ds.hpp"

#include <optional>
#include <vector>

struct ObstacleAvoidanceCandidate {
  Waypoint waypoint;
  int obstacle_index = -1;
};

std::optional<ObstacleAvoidanceCandidate> select_obstacle_detour_waypoint(
    double x,
    double y,
    double psi,
    const Waypoint& target_wp,
    const std::vector<CircleObstacle>& obstacles,
    const std::vector<bool>& skip_obstacles,
    double clearance,
    double trigger_margin,
    double detour_tol);
