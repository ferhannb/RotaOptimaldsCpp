#include "scenario_parser.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {
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

  if (s.waypoints.empty()) {
    throw std::runtime_error("Scenario must define at least one waypoint.");
  }

  return s;
}
