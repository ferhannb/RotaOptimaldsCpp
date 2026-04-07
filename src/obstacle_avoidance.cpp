#include "obstacle_avoidance.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {
double wrap_to_pi(double a) {
  return std::atan2(std::sin(a), std::cos(a));
}

double signed_turn_from_heading(double x, double y, double psi, double tx, double ty) {
  const double hx = std::cos(psi);
  const double hy = std::sin(psi);
  const double dx = tx - x;
  const double dy = ty - y;
  return hx * dy - hy * dx;
}

double score_candidate(double x, double y, double psi, double gx, double gy, double cx, double cy) {
  const double h = std::atan2(cy - y, cx - x);
  const double heading_cost = std::abs(wrap_to_pi(h - psi));
  const double goal_cost = std::hypot(cx - gx, cy - gy);
  return goal_cost + 0.2 * heading_cost;
}

bool segment_intersects_circle(
    double x0,
    double y0,
    double x1,
    double y1,
    double cx,
    double cy,
    double radius,
    double* t_closest) {
  const double vx = x1 - x0;
  const double vy = y1 - y0;
  const double v2 = vx * vx + vy * vy;
  if (v2 < 1e-12) {
    *t_closest = 0.0;
    return std::hypot(cx - x0, cy - y0) <= radius;
  }

  const double wx = cx - x0;
  const double wy = cy - y0;
  const double t_raw = (wx * vx + wy * vy) / v2;
  const double t = std::clamp(t_raw, 0.0, 1.0);
  *t_closest = t;
  const double px = x0 + t * vx;
  const double py = y0 + t * vy;
  return std::hypot(cx - px, cy - py) <= radius;
}
}  // namespace

std::optional<ObstacleAvoidanceCandidate> select_obstacle_detour_waypoint(
    double x,
    double y,
    double psi,
    const Waypoint& target_wp,
    const std::vector<CircleObstacle>& obstacles,
    const std::vector<bool>& skip_obstacles,
    double clearance,
    double trigger_margin,
    double detour_tol) {
  const double clearance_safe = std::max(0.0, clearance);
  const double trigger_safe = std::max(0.0, trigger_margin);

  int best_idx = -1;
  double best_t = std::numeric_limits<double>::infinity();
  Waypoint best_wp;

  for (int i = 0; i < static_cast<int>(obstacles.size()); ++i) {
    if (i < static_cast<int>(skip_obstacles.size()) && skip_obstacles[i]) {
      continue;
    }
    const CircleObstacle& ob = obstacles[i];
    if (!ob.enabled || ob.radius <= 0.0) {
      continue;
    }

    const double r_detour = ob.radius + clearance_safe;
    const double r_trigger = r_detour + trigger_safe;

    double t_closest = 0.0;
    if (!segment_intersects_circle(x, y, target_wp.X, target_wp.Y, ob.cx, ob.cy, r_trigger, &t_closest)) {
      continue;
    }

    const double vx = ob.cx - x;
    const double vy = ob.cy - y;
    const double vnorm = std::hypot(vx, vy);
    if (vnorm < 1e-9) {
      continue;
    }

    const double ux = vx / vnorm;
    const double uy = vy / vnorm;
    const double nx = -uy;
    const double ny = ux;

    const double c1x = ob.cx + r_detour * nx;
    const double c1y = ob.cy + r_detour * ny;
    const double c2x = ob.cx - r_detour * nx;
    const double c2y = ob.cy - r_detour * ny;

    const double s1 = score_candidate(x, y, psi, target_wp.X, target_wp.Y, c1x, c1y);
    const double s2 = score_candidate(x, y, psi, target_wp.X, target_wp.Y, c2x, c2y);
    const double turn1 = signed_turn_from_heading(x, y, psi, c1x, c1y);
    const double turn2 = signed_turn_from_heading(x, y, psi, c2x, c2y);

    Waypoint detour;
    double turn_sign = 0.0;
    if (s1 <= s2) {
      detour.X = c1x;
      detour.Y = c1y;
      turn_sign = turn1;
    } else {
      detour.X = c2x;
      detour.Y = c2y;
      turn_sign = turn2;
    }

    double kf_detour = 0.0;
    if (r_detour > 1e-9) {
      const double k_mag = 1.0 / r_detour;

      // Auto secimde aday WP score ile secilir; K isareti, rota WP'sinin
      // secilen detour WP'ye gore sancak/iskele tarafinda kalmasina gore verilir.
      const double psi_detour = std::atan2(detour.Y - y, detour.X - x);
      const double target_side = signed_turn_from_heading(
          detour.X, detour.Y, psi_detour, target_wp.X, target_wp.Y);

      if (std::abs(target_side) > 1e-9) {
        // Rota WP detour WP'nin sancağinda ise K negatif, iskelesinde ise pozitif.
        kf_detour = (target_side < 0.0) ? -k_mag : k_mag;
      } else {
        // Degenerate durumda onceki yon sinyalini kullan.
        kf_detour = (turn_sign >= 0.0) ? k_mag : -k_mag;
      }
    }

    // Set detour heading from circle tangent at this point.
    const double rx = detour.X - ob.cx;
    const double ry = detour.Y - ob.cy;
    const double radial_norm = std::hypot(rx, ry);
    if (radial_norm > 1e-9) {
      // +K => left/CCW tangent, -K => right/CW tangent
      double tx = (kf_detour >= 0.0) ? (-ry) : (ry);
      double ty = (kf_detour >= 0.0) ? (rx) : (-rx);

      // Choose tangent direction that points toward the main target waypoint.
      const double gx = target_wp.X - detour.X;
      const double gy = target_wp.Y - detour.Y;
      if ((tx * gx + ty * gy) < 0.0) {
        tx = -tx;
        ty = -ty;
      }
      detour.psig = std::atan2(ty, tx);
    } else {
      detour.psig = std::nullopt;
    }
    detour.Kf = kf_detour;
    detour.tol = std::max(0.1, detour_tol);
    detour.use_Kf = true;
    detour.w_wp = std::nullopt;
    detour.hit_scale = std::nullopt;

    if (t_closest < best_t) {
      best_t = t_closest;
      best_idx = i;
      best_wp = detour;
    }
  }

  if (best_idx < 0) {
    return std::nullopt;
  }

  return ObstacleAvoidanceCandidate{best_wp, best_idx};
}
