from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rota_optimal_ds import CircleObstacle, MPCConfig, RecedingOptions, State4, Waypoint


@dataclass
class ScenarioSpec:
    cfg: MPCConfig
    initial_state: State4
    opts: RecedingOptions
    waypoints: list[Waypoint]
    special_maneuver: str = "none"
    turn_dir: str = "auto"
    maneuver_radius: Optional[float] = None
    source: str = ""


def wrap_to_pi_np(a):
    return math.atan2(math.sin(a), math.cos(a))


def trim(s):
    return s.strip()


def lower(s):
    return trim(s).lower()


def split_csv(s):
    return [trim(item) for item in s.split(",")]


def scenario_dir_of(scenario_path):
    p = Path(scenario_path)
    return p.parent if p.parent != Path("") else Path(".")


def join_under_scenario_dir(scenario_path, file_path):
    p = Path(file_path)
    return str(p if p.is_absolute() else scenario_dir_of(scenario_path) / p)


def parse_bool(s):
    v = lower(s)
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"Invalid bool value: {s}")


def parse_double(s):
    return float(trim(s))


def parse_int(s):
    return int(trim(s))


def parse_opt_double(s):
    v = lower(s)
    if v in {"", "none", "nan", "null", "-"}:
        return None
    return parse_double(v)


def parse_int_list(s):
    v = lower(s)
    if v in {"", "none", "null", "empty", "-"}:
        return []
    out: list[int] = []
    for token in split_csv(s):
        if token:
            out.append(parse_int(token))
    return out


def make_waypoint(
    x,
    y,
    psig,
    Kf,
    tol,
    use_Kf,
    w_wp,
    hit_scale,
):
    return Waypoint(
        X=x,
        Y=y,
        psig=psig,
        Kf=Kf,
        tol=tol,
        use_Kf=use_Kf,
        w_wp=w_wp,
        hit_scale=hit_scale,
    )


def advance_arc(x, y, psi, radius, dpsi):
    if abs(dpsi) < 1e-12:
        return x, y, psi
    k = (1.0 if dpsi > 0.0 else -1.0) / radius
    psi1 = wrap_to_pi_np(psi + dpsi)
    x1 = x + (math.sin(psi + dpsi) - math.sin(psi)) / k
    y1 = y - (math.cos(psi + dpsi) - math.cos(psi)) / k
    return x1, y1, psi1


def build_williamson_waypoints(s):
    if len(s.waypoints) != 1:
        raise RuntimeError("Williamson maneuver requires exactly one final waypoint.")

    final_wp = s.waypoints[0]
    if final_wp.psig is None:
        raise RuntimeError("Williamson maneuver requires final waypoint heading (psig).")

    dpsi = wrap_to_pi_np(final_wp.psig - s.initial_state.psi)
    if abs(abs(dpsi) - math.pi) > 20.0 * math.pi / 180.0:
        raise RuntimeError("Williamson maneuver requires final heading to be approximately 180 degrees from the start heading.")

    if s.turn_dir in {"starboard", "right"}:
        turn_sign = -1
    elif s.turn_dir in {"port", "left"}:
        turn_sign = 1
    elif s.turn_dir == "auto":
        turn_sign = -1 if dpsi < 0.0 else 1
    else:
        raise RuntimeError(f"Invalid turn_dir: {s.turn_dir}")

    R_auto = max(2.0, 1.0 / max(s.cfg.K_MAX, 1e-6))
    R = s.maneuver_radius if s.maneuver_radius is not None else R_auto
    if R <= 0.0:
        raise RuntimeError("maneuver_radius must be > 0")

    psi0 = s.initial_state.psi
    ux = math.cos(psi0)
    uy = math.sin(psi0)
    tol_mid = max(3.0, 0.25 * R, final_wp.tol if final_wp.tol is not None else s.opts.tol_default)

    x1, y1, psi1 = advance_arc(
        s.initial_state.x,
        s.initial_state.y,
        psi0,
        R,
        turn_sign * 60.0 * math.pi / 180.0,
    )
    x2, y2, psi2 = advance_arc(x1, y1, psi1, R, -turn_sign * 220.0 * math.pi / 180.0)

    along2 = (x2 - s.initial_state.x) * ux + (y2 - s.initial_state.y) * uy
    return_margin = max(1.5 * R, 8.0)
    along3 = along2 - return_margin
    x3 = s.initial_state.x + along3 * ux
    y3 = s.initial_state.y + along3 * uy
    psi3 = wrap_to_pi_np(psi0 + math.pi)

    return [
        make_waypoint(x1, y1, psi1, 0.0, tol_mid, False, None, None),
        make_waypoint(x2, y2, psi2, 0.0, tol_mid, False, None, None),
        make_waypoint(x3, y3, psi3, 0.0, max(2.0, tol_mid), False, None, None),
    ]


def apply_special_maneuver(s):
    if not s.special_maneuver or s.special_maneuver == "none":
        return
    if s.special_maneuver == "williamson":
        s.waypoints = build_williamson_waypoints(s)
        return
    raise RuntimeError(f"Unknown special_maneuver: {s.special_maneuver}")


def parse_waypoint(value):
    toks = split_csv(value)
    if len(toks) < 2:
        raise RuntimeError("waypoint requires at least X,Y")

    wp = Waypoint()
    wp.X = parse_double(toks[0])
    wp.Y = parse_double(toks[1])
    if len(toks) > 2:
        wp.psig = parse_opt_double(toks[2])
    if len(toks) > 3:
        wp.Kf = parse_opt_double(toks[3])
    if len(toks) > 4:
        wp.tol = parse_opt_double(toks[4])
    if len(toks) > 5:
        wp.use_Kf = parse_bool(toks[5])
    if len(toks) > 6:
        wp.w_wp = parse_opt_double(toks[6])
    if len(toks) > 7:
        wp.hit_scale = parse_opt_double(toks[7])
    return wp


def parse_circle_obstacle(value):
    toks = split_csv(value)
    if len(toks) < 3:
        raise RuntimeError("obstacle requires cx,cy,radius")

    ob = CircleObstacle()
    ob.cx = parse_double(toks[0])
    ob.cy = parse_double(toks[1])
    ob.radius = parse_double(toks[2])
    if ob.radius <= 0.0:
        raise RuntimeError("obstacle radius must be > 0")
    if len(toks) > 3:
        ob.enabled = parse_bool(toks[3])
    return ob


def load_circle_obstacles_csv(path):
    if not Path(path).is_file():
        raise RuntimeError(f"Cannot open obstacles csv file: {path}")

    out: list[CircleObstacle] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = trim(raw.split("#", 1)[0])
            if not line:
                continue

            cols = split_csv(line)
            if len(cols) < 3:
                continue

            c0 = lower(cols[0])
            c1 = lower(cols[1])
            c2 = lower(cols[2])
            if (c0 in {"cx", "x"}) and (c1 in {"cy", "y"}) and (c2 in {"r", "radius"}):
                continue

            try:
                ob = CircleObstacle(
                    cx=parse_double(cols[0]),
                    cy=parse_double(cols[1]),
                    radius=parse_double(cols[2]),
                    enabled=True,
                )
                if ob.radius <= 0.0:
                    raise RuntimeError("radius must be > 0")
                if len(cols) > 3 and trim(cols[3]):
                    ob.enabled = parse_bool(cols[3])
            except Exception as exc:
                raise RuntimeError(f"{path}:{line_no} -> {exc}") from exc
            out.append(ob)
    return out


def set_key_value(s, key_in, value):
    key = lower(key_in)

    if key in {"n", "n_mpc"}:
        s.cfg.N = parse_int(value)
    elif key == "ds_min":
        s.cfg.ds_min = parse_double(value)
    elif key == "ds_max":
        s.cfg.ds_max = parse_double(value)
    elif key in {"k_max", "k_max_curvature"}:
        s.cfg.K_MAX = parse_double(value)
    elif key == "s_max":
        s.cfg.S_MAX = parse_double(value)
    elif key == "nseg":
        s.cfg.nseg = parse_int(value)
    elif key == "w_pos":
        s.cfg.w_pos = parse_double(value)
    elif key == "w_psi":
        s.cfg.w_psi = parse_double(value)
    elif key == "w_k":
        s.cfg.w_K = parse_double(value)
    elif key == "w_kcmd":
        s.cfg.w_Kcmd = parse_double(value)
    elif key == "w_dkcmd":
        s.cfg.w_dKcmd = parse_double(value)
    elif key == "w_ds_smooth":
        s.cfg.w_ds_smooth = parse_double(value)
    elif key == "ds_jump_max":
        s.cfg.ds_jump_max = parse_opt_double(value)
    elif key == "w_kf":
        s.cfg.w_Kf = parse_double(value)
    elif key == "enable_terminal_k_hard":
        s.cfg.enable_terminal_K_hard = parse_bool(value)
    elif key == "ipopt_max_iter":
        s.cfg.ipopt_max_iter = parse_int(value)
    elif key == "ipopt_tol":
        s.cfg.ipopt_tol = parse_double(value)
    elif key == "block_lengths_kcmd":
        s.cfg.block_lengths_Kcmd = parse_int_list(value)
    elif key == "block_lengths_ds":
        s.cfg.block_lengths_ds = parse_int_list(value)
    elif key == "w_prog":
        s.cfg.w_prog = parse_double(value)
    elif key == "alpha_prog":
        s.cfg.alpha_prog = parse_double(value)
    elif key == "hit_ratio":
        s.cfg.hit_ratio = parse_double(value)
    elif key == "x0":
        s.initial_state.x = parse_double(value)
    elif key == "y0":
        s.initial_state.y = parse_double(value)
    elif key == "psi0":
        s.initial_state.psi = parse_double(value)
    elif key == "k0":
        s.initial_state.K = parse_double(value)
    elif key == "tol_default":
        s.opts.tol_default = parse_double(value)
    elif key == "max_iters":
        s.opts.max_iters = parse_int(value)
    elif key == "use_heading_gate":
        s.opts.use_heading_gate = parse_bool(value)
    elif key == "tol_psi":
        s.opts.tol_psi = parse_double(value)
    elif key == "tol_psi_deg":
        s.opts.tol_psi = parse_double(value) * math.pi / 180.0
    elif key == "w_wp_intermediate":
        s.opts.w_wp_intermediate = parse_double(value)
    elif key == "term_scale_intermediate":
        s.opts.term_scale_intermediate = parse_double(value)
    elif key == "term_scale_final":
        s.opts.term_scale_final = parse_double(value)
    elif key == "hit_scale_intermediate":
        s.opts.hit_scale_intermediate = parse_double(value)
    elif key == "w_wp_final":
        s.opts.w_wp_final = parse_double(value)
    elif key == "use_wp_kf":
        s.opts.use_wp_kf = parse_bool(value)
    elif key == "kf_fallback":
        s.opts.kf_fallback = parse_double(value)
    elif key == "enable_obstacle_avoidance":
        s.opts.enable_obstacle_avoidance = parse_bool(value)
    elif key == "obstacle_clearance":
        s.opts.obstacle_clearance = parse_double(value)
    elif key == "obstacle_trigger_margin":
        s.opts.obstacle_trigger_margin = parse_double(value)
    elif key == "obstacle_waypoint_tol":
        s.opts.obstacle_waypoint_tol = parse_double(value)
    elif key in {"special_maneuver", "maneuver"}:
        s.special_maneuver = lower(value)
    elif key in {"turn_dir", "turn_direction"}:
        s.turn_dir = lower(value)
    elif key in {"maneuver_radius", "williamson_radius"}:
        s.maneuver_radius = parse_double(value)
    else:
        raise RuntimeError(f"Unknown key in scenario file: {key}")


def make_default_scenario():
    s = ScenarioSpec(
        cfg=MPCConfig(),
        initial_state=State4(0.0, 0.0, 0.0, 0.0),
        opts=RecedingOptions(),
        waypoints=[],
        source="built-in defaults",
    )

    s.cfg.N = 20
    s.cfg.w_pos = 50.0
    s.cfg.w_psi = 50.0
    s.cfg.w_K = 1.0
    s.cfg.w_Kcmd = 0.25
    s.cfg.w_dKcmd = 15.0
    s.cfg.w_ds_smooth = 0.01
    s.cfg.ds_jump_max = 0.0
    s.cfg.w_Kf = 150.0
    s.cfg.ds_max = 2.0
    s.cfg.K_MAX = 0.3
    s.cfg.block_lengths_Kcmd = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    s.cfg.block_lengths_ds = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    s.cfg.w_prog = 0.0
    s.cfg.alpha_prog = 0.0
    s.cfg.hit_ratio = 0.7

    s.waypoints.append(Waypoint(30.0, 20.0, 0.0, 0.1, 0.5, None, None, True))
    s.waypoints.append(Waypoint(60.0, 0.0, 0.0, -0.1, 0.5, None, None, True))

    s.opts.use_heading_gate = True
    s.opts.tol_psi = 12.0 * 3.14 / 180.0
    s.opts.w_wp_intermediate = 5.0
    s.opts.term_scale_intermediate = 0.2
    s.opts.term_scale_final = 1.0
    s.opts.hit_scale_intermediate = 0.7
    s.opts.w_wp_final = 1.0
    s.opts.use_wp_kf = True
    s.opts.kf_fallback = 0.0

    return s


def load_scenario_ini(path):
    scenario_path = Path(path)
    if not scenario_path.is_file():
        raise RuntimeError(f"Cannot open scenario file: {path}")

    s = make_default_scenario()
    s.source = path

    saw_waypoint = False
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = trim(raw.split("#", 1)[0])
            if not line:
                continue
            if "=" not in line:
                raise RuntimeError(f"Invalid line (missing '=') at {path}:{line_no}")

            key, value = [trim(part) for part in line.split("=", 1)]
            key_l = lower(key)

            if key_l in {"waypoint", "wp"}:
                if not saw_waypoint:
                    s.waypoints.clear()
                    saw_waypoint = True
                s.waypoints.append(parse_waypoint(value))
                continue

            if key_l in {"obstacle", "circle_obstacle"}:
                s.opts.obstacles.append(parse_circle_obstacle(value))
                continue

            if key_l == "obstacles_csv":
                csv_path = join_under_scenario_dir(path, value)
                s.opts.obstacles.extend(load_circle_obstacles_csv(csv_path))
                continue

            try:
                set_key_value(s, key, value)
            except Exception as exc:
                raise RuntimeError(f"{path}:{line_no} -> {exc}") from exc

    if not s.waypoints:
        raise RuntimeError("Scenario must define at least one waypoint.")

    apply_special_maneuver(s)
    return s
