#include "colreg_scenarios.hpp"
#include "scenario_parser.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {
double wrap_to_pi_np(double a) {
  return std::atan2(std::sin(a), std::cos(a));
}

std::string trim(const std::string& s) {
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) {
    return "";
  }
  const auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    out.push_back(trim(item));
  }
  return out;
}

std::string scenario_dir_of(const std::string& scenario_path) {
  std::filesystem::path p(scenario_path);
  if (p.has_parent_path()) {
    return p.parent_path().string();
  }
  return ".";
}

std::string join_under_scenario_dir(const std::string& scenario_path, const std::string& file_path) {
  std::filesystem::path p(file_path);
  if (p.is_absolute()) {
    return p.string();
  }
  return (std::filesystem::path(scenario_dir_of(scenario_path)) / p).string();
}

bool parse_bool(const std::string& s) {
  const std::string v = lower(trim(s));
  if (v == "1" || v == "true" || v == "yes" || v == "on") {
    return true;
  }
  if (v == "0" || v == "false" || v == "no" || v == "off") {
    return false;
  }
  throw std::runtime_error("Invalid bool value: " + s);
}

double parse_double(const std::string& s) {
  return std::stod(trim(s));
}

int parse_int(const std::string& s) {
  return std::stoi(trim(s));
}

std::optional<double> parse_opt_double(const std::string& s) {
  const std::string v = lower(trim(s));
  if (v.empty() || v == "none" || v == "nan" || v == "null" || v == "-") {
    return std::nullopt;
  }
  return parse_double(v);
}

std::vector<int> parse_int_list(const std::string& s) {
  const std::string v = lower(trim(s));
  if (v.empty() || v == "none" || v == "null" || v == "empty" || v == "-") {
    return {};
  }
  const auto toks = split_csv(s);
  std::vector<int> out;
  out.reserve(toks.size());
  for (const auto& t : toks) {
    if (!t.empty()) {
      out.push_back(parse_int(t));
    }
  }
  return out;
}

Waypoint make_waypoint(
    double x,
    double y,
    std::optional<double> psig,
    std::optional<double> Kf,
    std::optional<double> tol,
    bool use_Kf,
    std::optional<double> w_wp,
    std::optional<double> hit_scale) {
  Waypoint wp;
  wp.X = x;
  wp.Y = y;
  wp.psig = psig;
  wp.Kf = Kf;
  wp.tol = tol;
  wp.use_Kf = use_Kf;
  wp.w_wp = w_wp;
  wp.hit_scale = hit_scale;
  return wp;
}

std::tuple<double, double, double> advance_arc(
    double x,
    double y,
    double psi,
    double radius,
    double dpsi) {
  if (std::abs(dpsi) < 1e-12) {
    return {x, y, psi};
  }
  const double k = ((dpsi > 0.0) ? 1.0 : -1.0) / radius;
  const double psi1 = wrap_to_pi_np(psi + dpsi);
  const double x1 = x + (std::sin(psi + dpsi) - std::sin(psi)) / k;
  const double y1 = y - (std::cos(psi + dpsi) - std::cos(psi)) / k;
  return {x1, y1, psi1};
}

std::vector<Waypoint> build_williamson_waypoints(const ScenarioSpec& s) {
  if (s.waypoints.size() != 1) {
    throw std::runtime_error("Williamson maneuver requires exactly one final waypoint.");
  }

  const Waypoint& final_wp = s.waypoints.front();
  if (!final_wp.psig.has_value()) {
    throw std::runtime_error("Williamson maneuver requires final waypoint heading (psig).");
  }

  const double dpsi = wrap_to_pi_np(*final_wp.psig - s.initial_state.psi);
  if (std::abs(std::abs(dpsi) - M_PI) > 20.0 * M_PI / 180.0) {
    throw std::runtime_error("Williamson maneuver requires final heading to be approximately 180 degrees from the start heading.");
  }

  int turn_sign = 1;
  if (s.turn_dir == "starboard" || s.turn_dir == "right") {
    turn_sign = -1;
  } else if (s.turn_dir == "port" || s.turn_dir == "left") {
    turn_sign = 1;
  } else if (s.turn_dir == "auto") {
    turn_sign = (dpsi < 0.0) ? -1 : 1;
  } else {
    throw std::runtime_error("Invalid turn_dir: " + s.turn_dir);
  }

  const double R_auto = std::max(2.0, 1.0 / std::max(s.cfg.K_MAX, 1e-6));
  const double R = s.maneuver_radius.has_value() ? *s.maneuver_radius : R_auto;
  if (R <= 0.0) {
    throw std::runtime_error("maneuver_radius must be > 0");
  }

  const double psi0 = s.initial_state.psi;
  const double ux = std::cos(psi0);
  const double uy = std::sin(psi0);
  const double tol_mid = std::max({3.0, 0.25 * R, final_wp.tol.value_or(s.opts.tol_default)});

  double x1, y1, psi1;
  std::tie(x1, y1, psi1) = advance_arc(
      s.initial_state.x,
      s.initial_state.y,
      psi0,
      R,
      turn_sign * 60.0 * M_PI / 180.0);

  double x2, y2, psi2;
  std::tie(x2, y2, psi2) = advance_arc(
      x1,
      y1,
      psi1,
      R,
      -turn_sign * 220.0 * M_PI / 180.0);

  const double along2 =
      (x2 - s.initial_state.x) * ux +
      (y2 - s.initial_state.y) * uy;
  const double return_margin = std::max(1.5 * R, 8.0);
  const double along3 = along2 - return_margin;
  const double x3 = s.initial_state.x + along3 * ux;
  const double y3 = s.initial_state.y + along3 * uy;
  const double psi3 = wrap_to_pi_np(psi0 + M_PI);

  std::vector<Waypoint> out;
  out.reserve(3);
  out.push_back(make_waypoint(x1, y1, psi1, 0.0, tol_mid, false, std::nullopt, std::nullopt));
  out.push_back(make_waypoint(x2, y2, psi2, 0.0, tol_mid, false, std::nullopt, std::nullopt));
  out.push_back(make_waypoint(x3, y3, psi3, 0.0, std::max(2.0, tol_mid), false, std::nullopt, std::nullopt));
  return out;
}

void apply_special_maneuver(ScenarioSpec* s) {
  if (s->special_maneuver.empty() || s->special_maneuver == "none") {
    return;
  }
  if (s->special_maneuver == "williamson") {
    s->waypoints = build_williamson_waypoints(*s);
    return;
  }
  throw std::runtime_error("Unknown special_maneuver: " + s->special_maneuver);
}

Waypoint parse_waypoint(const std::string& value) {
  const auto t = split_csv(value);
  if (t.size() < 2) {
    throw std::runtime_error("waypoint requires at least X,Y");
  }

  Waypoint wp;
  wp.X = parse_double(t[0]);
  wp.Y = parse_double(t[1]);
  if (t.size() > 2) wp.psig = parse_opt_double(t[2]);
  if (t.size() > 3) wp.Kf = parse_opt_double(t[3]);
  if (t.size() > 4) wp.tol = parse_opt_double(t[4]);
  if (t.size() > 5) wp.use_Kf = parse_bool(t[5]);
  if (t.size() > 6) wp.w_wp = parse_opt_double(t[6]);
  if (t.size() > 7) wp.hit_scale = parse_opt_double(t[7]);
  return wp;
}

ShipKinematics parse_ship_kinematics(const std::vector<std::string>& t, int start_idx) {
  if (static_cast<int>(t.size()) < start_idx + 4) {
    throw std::runtime_error("ship requires x,y,course_deg,speed");
  }

  ShipKinematics kin;
  kin.x = parse_double(t[start_idx + 0]);
  kin.y = parse_double(t[start_idx + 1]);
  kin.course = parse_double(t[start_idx + 2]) * M_PI / 180.0;
  kin.speed = parse_double(t[start_idx + 3]);
  return kin;
}

OwnShip parse_own_ship(const std::string& value) {
  const auto t = split_csv(value);
  if (t.size() < 5) {
    throw std::runtime_error("own_ship requires name,x,y,course_deg,speed[,length,beam]");
  }

  OwnShip ship;
  ship.name = t[0];
  ship.state = parse_ship_kinematics(t, 1);
  if (t.size() > 5) ship.length = parse_double(t[5]);
  if (t.size() > 6) ship.beam = parse_double(t[6]);
  return ship;
}

TargetShip parse_target_ship(const std::string& value) {
  const auto t = split_csv(value);
  if (t.size() < 5) {
    throw std::runtime_error("target_ship requires id,x,y,course_deg,speed[,length,beam]");
  }

  TargetShip ship;
  ship.id = t[0];
  ship.state = parse_ship_kinematics(t, 1);
  if (t.size() > 5) ship.length = parse_double(t[5]);
  if (t.size() > 6) ship.beam = parse_double(t[6]);
  return ship;
}

CircleObstacle parse_circle_obstacle(const std::string& value) {
  const auto t = split_csv(value);
  if (t.size() < 3) {
    throw std::runtime_error("obstacle requires cx,cy,radius");
  }

  CircleObstacle ob;
  ob.cx = parse_double(t[0]);
  ob.cy = parse_double(t[1]);
  ob.radius = parse_double(t[2]);
  if (ob.radius <= 0.0) {
    throw std::runtime_error("obstacle radius must be > 0");
  }
  if (t.size() > 3) {
    ob.enabled = parse_bool(t[3]);
  }
  return ob;
}

std::vector<CircleObstacle> load_circle_obstacles_csv(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    throw std::runtime_error("Cannot open obstacles csv file: " + path);
  }

  std::vector<CircleObstacle> out;
  std::string line;
  int line_no = 0;
  while (std::getline(ifs, line)) {
    ++line_no;
    const auto h = line.find('#');
    if (h != std::string::npos) {
      line = line.substr(0, h);
    }
    line = trim(line);
    if (line.empty()) {
      continue;
    }

    const auto cols = split_csv(line);
    if (cols.size() < 3) {
      continue;
    }

    const std::string c0 = lower(cols[0]);
    const std::string c1 = lower(cols[1]);
    const std::string c2 = lower(cols[2]);
    if ((c0 == "cx" || c0 == "x") && (c1 == "cy" || c1 == "y") &&
        (c2 == "r" || c2 == "radius")) {
      continue;
    }

    CircleObstacle ob;
    try {
      ob.cx = parse_double(cols[0]);
      ob.cy = parse_double(cols[1]);
      ob.radius = parse_double(cols[2]);
      if (ob.radius <= 0.0) {
        throw std::runtime_error("radius must be > 0");
      }
      if (cols.size() > 3 && !trim(cols[3]).empty()) {
        ob.enabled = parse_bool(cols[3]);
      }
    } catch (const std::exception& e) {
      throw std::runtime_error(path + ":" + std::to_string(line_no) + " -> " + e.what());
    }
    out.push_back(ob);
  }

  return out;
}

void set_key_value(ScenarioSpec* s, const std::string& key_in, const std::string& value) {
  const std::string key = lower(trim(key_in));

  if (key == "n" || key == "n_mpc") s->cfg.N = parse_int(value);
  else if (key == "ds_min") s->cfg.ds_min = parse_double(value);
  else if (key == "ds_max") s->cfg.ds_max = parse_double(value);
  else if (key == "k_max" || key == "k_max_curvature") s->cfg.K_MAX = parse_double(value);
  else if (key == "s_max") s->cfg.S_MAX = parse_double(value);
  else if (key == "nseg") s->cfg.nseg = parse_int(value);
  else if (key == "w_pos") s->cfg.w_pos = parse_double(value);
  else if (key == "w_psi") s->cfg.w_psi = parse_double(value);
  else if (key == "w_k") s->cfg.w_K = parse_double(value);
  else if (key == "w_kcmd") s->cfg.w_Kcmd = parse_double(value);
  else if (key == "w_dkcmd") s->cfg.w_dKcmd = parse_double(value);
  else if (key == "w_ds_smooth") s->cfg.w_ds_smooth = parse_double(value);
  else if (key == "ds_jump_max") {
    auto v = parse_opt_double(value);
    s->cfg.ds_jump_max = v;
  }
  else if (key == "w_kf") s->cfg.w_Kf = parse_double(value);
  else if (key == "enable_terminal_k_hard") s->cfg.enable_terminal_K_hard = parse_bool(value);
  else if (key == "ipopt_max_iter") s->cfg.ipopt_max_iter = parse_int(value);
  else if (key == "ipopt_tol") s->cfg.ipopt_tol = parse_double(value);
  else if (key == "nlp_solver" || key == "solver") s->cfg.nlp_solver = lower(trim(value));
  else if (key == "realtime_mode") s->cfg.realtime_mode = parse_bool(value);
  else if (key == "fatrop_structure_detection") s->cfg.fatrop_structure_detection = lower(trim(value));
  else if (key == "fatrop_debug") s->cfg.fatrop_debug = parse_bool(value);
  else if (key == "fatrop_convexify_strategy") s->cfg.fatrop_convexify_strategy = lower(trim(value));
  else if (key == "fatrop_convexify_margin") s->cfg.fatrop_convexify_margin = parse_double(value);
  else if (key == "block_lengths_kcmd") s->cfg.block_lengths_Kcmd = parse_int_list(value);
  else if (key == "block_lengths_ds") s->cfg.block_lengths_ds = parse_int_list(value);
  else if (key == "w_prog") s->cfg.w_prog = parse_double(value);
  else if (key == "alpha_prog") s->cfg.alpha_prog = parse_double(value);
  else if (key == "hit_ratio") s->cfg.hit_ratio = parse_double(value);

  else if (key == "x0") s->initial_state.x = parse_double(value);
  else if (key == "y0") s->initial_state.y = parse_double(value);
  else if (key == "psi0") s->initial_state.psi = parse_double(value);
  else if (key == "k0") s->initial_state.K = parse_double(value);

  else if (key == "tol_default") s->opts.tol_default = parse_double(value);
  else if (key == "max_iters") s->opts.max_iters = parse_int(value);
  else if (key == "use_heading_gate") s->opts.use_heading_gate = parse_bool(value);
  else if (key == "tol_psi") s->opts.tol_psi = parse_double(value);
  else if (key == "tol_psi_deg") s->opts.tol_psi = parse_double(value) * 3.14159265358979323846 / 180.0;
  else if (key == "w_wp_intermediate") s->opts.w_wp_intermediate = parse_double(value);
  else if (key == "term_scale_intermediate") s->opts.term_scale_intermediate = parse_double(value);
  else if (key == "term_scale_final") s->opts.term_scale_final = parse_double(value);
  else if (key == "hit_scale_intermediate") s->opts.hit_scale_intermediate = parse_double(value);
  else if (key == "w_wp_final") s->opts.w_wp_final = parse_double(value);
  else if (key == "use_wp_kf") s->opts.use_wp_kf = parse_bool(value);
  else if (key == "kf_fallback") s->opts.kf_fallback = parse_double(value);
  else if (key == "enable_obstacle_avoidance") s->opts.enable_obstacle_avoidance = parse_bool(value);
  else if (key == "obstacle_clearance") s->opts.obstacle_clearance = parse_double(value);
  else if (key == "obstacle_trigger_margin") s->opts.obstacle_trigger_margin = parse_double(value);
  else if (key == "obstacle_waypoint_tol") s->opts.obstacle_waypoint_tol = parse_double(value);
  else if (key == "colreg_only") s->colreg_only = parse_bool(value);
  else if (key == "colreg_risk_dcpa") s->colreg_cfg.risk_dcpa = parse_double(value);
  else if (key == "colreg_max_tcpa") s->colreg_cfg.max_tcpa = parse_double(value);
  else if (key == "colreg_alpha_crit_13_deg") s->colreg_cfg.alpha_crit_13_deg = parse_double(value);
  else if (key == "colreg_alpha_crit_14_deg") s->colreg_cfg.alpha_crit_14_deg = parse_double(value);
  else if (key == "colreg_alpha_crit_15_deg") s->colreg_cfg.alpha_crit_15_deg = parse_double(value);
  else if (key == "colreg_overtaking_sector_deg") s->colreg_cfg.overtaking_sector_deg = parse_double(value);
  else if (key == "colreg_crossing_sector_deg") s->colreg_cfg.crossing_sector_deg = parse_double(value);
  else if (key == "special_maneuver" || key == "maneuver") s->special_maneuver = lower(trim(value));
  else if (key == "turn_dir" || key == "turn_direction") s->turn_dir = lower(trim(value));
  else if (key == "maneuver_radius" || key == "williamson_radius") s->maneuver_radius = parse_double(value);

  else {
    throw std::runtime_error("Unknown key in scenario file: " + key);
  }
}
}  // namespace

ScenarioSpec make_default_scenario() {
  ScenarioSpec s;
  s.source = "built-in defaults";

  s.cfg.N       = 20;
  s.cfg.w_pos   = 50.0;
  s.cfg.w_psi   = 50.0;
  s.cfg.w_K     = 1.0;
  s.cfg.w_Kcmd  = 0.25;
  s.cfg.w_dKcmd = 15.0;
  s.cfg.w_ds_smooth = 0.01;
  s.cfg.ds_jump_max = 0.0;
  s.cfg.w_Kf = 150.0;
  s.cfg.ds_max = 2.0;
  s.cfg.K_MAX = 0.3;
  s.cfg.block_lengths_Kcmd = {1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  s.cfg.block_lengths_ds = {1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  s.cfg.w_prog = 0.0;
  s.cfg.alpha_prog = 0.0;
  s.cfg.hit_ratio = 0.7;

  s.initial_state = State4{0.0, 0.0, 0.0, 0.0};

  s.waypoints.push_back(Waypoint{30.0, 20.0, 0.0, 0.1, 0.5, std::nullopt, std::nullopt, true});
  s.waypoints.push_back(Waypoint{60.0, 00.0, 0.0, -0.1, 0.5, std::nullopt, std::nullopt, true});

  s.opts.use_heading_gate = true;
  s.opts.tol_psi = 12.0 * 3.14 / 180.0;
  s.opts.w_wp_intermediate = 5.0;
  s.opts.term_scale_intermediate = 0.2;
  s.opts.term_scale_final = 1.0;
  s.opts.hit_scale_intermediate = 0.7;
  s.opts.w_wp_final = 1.0;
  s.opts.use_wp_kf = true;
  s.opts.kf_fallback = 0.0;
  s.colreg_cfg.risk_dcpa = 0.5;
  s.colreg_cfg.max_tcpa = 20.0;
  s.colreg_cfg.alpha_crit_13_deg = 45.0;
  s.colreg_cfg.alpha_crit_14_deg = 13.0;
  s.colreg_cfg.alpha_crit_15_deg = 10.0;
  s.colreg_cfg.overtaking_sector_deg = 112.5;
  s.colreg_cfg.crossing_sector_deg = 112.5;

  return s;
}

ScenarioSpec load_scenario_ini(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    throw std::runtime_error("Cannot open scenario file: " + path);
  }

  ScenarioSpec s = make_default_scenario();
  s.source = path;

  std::string line;
  int line_no = 0;
  bool saw_waypoint = false;
  bool saw_colreg_preset = false;
  bool saw_explicit_colreg_ships = false;

  while (std::getline(ifs, line)) {
    ++line_no;
    const auto h = line.find('#');
    if (h != std::string::npos) {
      line = line.substr(0, h);
    }
    line = trim(line);
    if (line.empty()) {
      continue;
    }

    const auto eq = line.find('=');
    if (eq == std::string::npos) {
      throw std::runtime_error("Invalid line (missing '=') at " + path + ":" + std::to_string(line_no));
    }

    const std::string key = trim(line.substr(0, eq));
    const std::string value = trim(line.substr(eq + 1));
    const std::string key_l = lower(key);

    if (key_l == "waypoint" || key_l == "wp") {
      if (!saw_waypoint) {
        s.waypoints.clear();
        saw_waypoint = true;
      }
      s.waypoints.push_back(parse_waypoint(value));
      continue;
    }
    if (key_l == "colreg_scenario" || key_l == "colreg_preset") {
      if (saw_explicit_colreg_ships) {
        throw std::runtime_error(
            path + ":" + std::to_string(line_no) +
            " -> colreg_scenario cannot be combined with explicit own_ship/target_ship entries.");
      }
      const EncounterScenario preset = make_colreg_scenario_by_name(lower(trim(value)));
      s.colreg_only = true;
      s.own_ship = preset.own_ship;
      s.target_ships.clear();
      s.target_ships.push_back(preset.target_ship);
      saw_colreg_preset = true;
      continue;
    }
    if (key_l == "own_ship") {
      if (saw_colreg_preset) {
        throw std::runtime_error(
            path + ":" + std::to_string(line_no) +
            " -> own_ship cannot be combined with colreg_scenario preset selection.");
      }
      s.own_ship = parse_own_ship(value);
      saw_explicit_colreg_ships = true;
      continue;
    }
    if (key_l == "target_ship") {
      if (saw_colreg_preset) {
        throw std::runtime_error(
            path + ":" + std::to_string(line_no) +
            " -> target_ship cannot be combined with colreg_scenario preset selection.");
      }
      s.target_ships.push_back(parse_target_ship(value));
      saw_explicit_colreg_ships = true;
      continue;
    }
    if (key_l == "obstacle" || key_l == "circle_obstacle") {
      s.opts.obstacles.push_back(parse_circle_obstacle(value));
      continue;
    }
    if (key_l == "obstacles_csv") {
      const std::string csv_path = join_under_scenario_dir(path, value);
      const auto obs = load_circle_obstacles_csv(csv_path);
      s.opts.obstacles.insert(s.opts.obstacles.end(), obs.begin(), obs.end());
      continue;
    }

    try {
      set_key_value(&s, key, value);
    } catch (const std::exception& e) {
      throw std::runtime_error(path + ":" + std::to_string(line_no) + " -> " + e.what());
    }
  }

  if (s.waypoints.empty() && !s.colreg_only) {
    throw std::runtime_error("Scenario must define at least one waypoint.");
  }
  if (s.colreg_only && !s.own_ship.has_value()) {
    throw std::runtime_error("COLREG-only scenario must define own_ship.");
  }
  if (s.colreg_only && s.target_ships.empty()) {
    throw std::runtime_error("COLREG-only scenario must define at least one target_ship.");
  }

  apply_special_maneuver(&s);

  return s;
}
