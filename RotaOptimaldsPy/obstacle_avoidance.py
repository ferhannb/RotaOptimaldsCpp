from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from rota_optimal_ds import CircleObstacle, Waypoint


@dataclass
class ObstacleAvoidanceCandidate:
    waypoint: Waypoint
    obstacle_index: int = -1


def wrap_to_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


def signed_turn_from_heading(x, y, psi, tx, ty):
    hx = math.cos(psi)
    hy = math.sin(psi)
    dx = tx - x
    dy = ty - y
    return hx * dy - hy * dx


def score_candidate(x, y, psi, gx, gy, cx, cy):
    h = math.atan2(cy - y, cx - x)
    heading_cost = abs(wrap_to_pi(h - psi))
    goal_cost = math.hypot(cx - gx, cy - gy)
    return goal_cost + 0.2 * heading_cost


def segment_intersects_circle(
    x0,
    y0,
    x1,
    y1,
    cx,
    cy,
    radius,
):
    vx = x1 - x0
    vy = y1 - y0
    v2 = vx * vx + vy * vy
    if v2 < 1e-12:
        return math.hypot(cx - x0, cy - y0) <= radius, 0.0

    wx = cx - x0
    wy = cy - y0
    t_raw = (wx * vx + wy * vy) / v2
    t = max(0.0, min(1.0, t_raw))
    px = x0 + t * vx
    py = y0 + t * vy
    return math.hypot(cx - px, cy - py) <= radius, t


def select_obstacle_detour_waypoint(
    x,
    y,
    psi,
    target_wp,
    obstacles,
    skip_obstacles,
    clearance,
    trigger_margin,
    detour_tol,
):
    clearance_safe = max(0.0, clearance)
    trigger_safe = max(0.0, trigger_margin)

    best_idx = -1
    best_t = math.inf
    best_wp: Optional[Waypoint] = None

    for i, ob in enumerate(obstacles):
        if i < len(skip_obstacles) and skip_obstacles[i]:
            continue
        if (not ob.enabled) or ob.radius <= 0.0:
            continue

        r_detour = ob.radius + clearance_safe
        r_trigger = r_detour + trigger_safe
        intersects, t_closest = segment_intersects_circle(
            x,
            y,
            target_wp.X,
            target_wp.Y,
            ob.cx,
            ob.cy,
            r_trigger,
        )
        if not intersects:
            continue

        vx = ob.cx - x
        vy = ob.cy - y
        vnorm = math.hypot(vx, vy)
        if vnorm < 1e-9:
            continue

        ux = vx / vnorm
        uy = vy / vnorm
        nx = -uy
        ny = ux

        c1x = ob.cx + r_detour * nx
        c1y = ob.cy + r_detour * ny
        c2x = ob.cx - r_detour * nx
        c2y = ob.cy - r_detour * ny

        s1 = score_candidate(x, y, psi, target_wp.X, target_wp.Y, c1x, c1y)
        s2 = score_candidate(x, y, psi, target_wp.X, target_wp.Y, c2x, c2y)
        turn1 = signed_turn_from_heading(x, y, psi, c1x, c1y)
        turn2 = signed_turn_from_heading(x, y, psi, c2x, c2y)

        detour = Waypoint()
        if s1 <= s2:
            detour.X = c1x
            detour.Y = c1y
            turn_sign = turn1
        else:
            detour.X = c2x
            detour.Y = c2y
            turn_sign = turn2

        kf_detour = 0.0
        if r_detour > 1e-9:
            k_mag = 1.0 / r_detour
            psi_detour = math.atan2(detour.Y - y, detour.X - x)
            target_side = signed_turn_from_heading(detour.X, detour.Y, psi_detour, target_wp.X, target_wp.Y)
            if abs(target_side) > 1e-9:
                kf_detour = -k_mag if target_side < 0.0 else k_mag
            else:
                kf_detour = k_mag if turn_sign >= 0.0 else -k_mag

        rx = detour.X - ob.cx
        ry = detour.Y - ob.cy
        radial_norm = math.hypot(rx, ry)
        if radial_norm > 1e-9:
            tx = -ry if kf_detour >= 0.0 else ry
            ty = rx if kf_detour >= 0.0 else -rx
            gx = target_wp.X - detour.X
            gy = target_wp.Y - detour.Y
            if (tx * gx + ty * gy) < 0.0:
                tx = -tx
                ty = -ty
            detour.psig = math.atan2(ty, tx)
        else:
            detour.psig = None

        detour.Kf = kf_detour
        detour.tol = max(0.1, detour_tol)
        detour.use_Kf = True
        detour.w_wp = None
        detour.hit_scale = None

        if t_closest < best_t:
            best_t = t_closest
            best_idx = i
            best_wp = detour

    if best_idx < 0 or best_wp is None:
        return None
    return ObstacleAvoidanceCandidate(best_wp, best_idx)
