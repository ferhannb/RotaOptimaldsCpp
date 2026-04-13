#include "colreg_classifier.hpp"
#include "rota_optimal_ds.hpp"
#include "scenario_parser.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
void print_help(const char* bin) {
  std::cout
      << "Usage: " << bin << " [options]\n"
      << "Options:\n"
      << "  --scenario <file>   Load scenario ini file\n" 
      << "  --out-log <file>    Output receding log csv (default: receding_log.csv)\n"
      << "  --out-wp <file>     Output waypoint csv (default: waypoints.csv)\n"
      << "  --colreg-scan       Run constant-velocity COLREG scan instead of MPC\n"
      << "  --out-colreg-log <file>  Output COLREG scan csv (default: colreg_scan.csv)\n"
      << "  --scan-dt <value>   COLREG scan time step (default: 0.5)\n"
      << "  --scan-steps <int>  COLREG scan steps (default: 80)\n"
      << "  --help              Show this help\n";
}

ShipKinematics propagate_const_velocity(const ShipKinematics& state, double dt) {
  ShipKinematics next = state;
  next.x += state.speed * std::cos(state.course) * dt;
  next.y += state.speed * std::sin(state.course) * dt;
  return next;
}

struct EncounterTrackState {
  EncounterType tracked_type = EncounterType::None;
  ColregRole tracked_role = ColregRole::None;
  bool engaged = false;
};

double current_range(const OwnShip& own, const TargetShip& target) {
  return std::hypot(target.state.x - own.state.x, target.state.y - own.state.y);
}

void update_track_state(
    const ColregEncounter& encounter,
    double range_now,
    double clear_range,
    EncounterTrackState* state) {
  const bool concrete_geometry =
      encounter.type != EncounterType::SafePassing &&
      encounter.type != EncounterType::Undefined &&
      encounter.type != EncounterType::None;

  if (concrete_geometry) {
    state->tracked_type = encounter.type;
    state->tracked_role = encounter.own_role;
    state->engaged = true;
    return;
  }

  if (!state->engaged) {
    state->tracked_type = encounter.type;
    state->tracked_role = encounter.own_role;
    return;
  }

  const bool cleared =
      (!encounter.collision_risk) &&
      (encounter.cpa.tcpa < 0.0) &&
      (range_now >= clear_range);
  if (cleared) {
    state->tracked_type = EncounterType::SafePassing;
    state->tracked_role = ColregRole::None;
    state->engaged = false;
  }
}

void print_colreg_report(const ScenarioSpec& scenario) {
  if (!scenario.own_ship.has_value() || scenario.target_ships.empty()) {
    return;
  }

  for (const auto& target : scenario.target_ships) {
    const ColregEncounter encounter =
        classify_encounter(*scenario.own_ship, target, scenario.colreg_cfg);
    std::cout << "COLREG target=" << target.id
              << " type=" << to_string(encounter.type)
              << " role=" << to_string(encounter.own_role)
              << " risk=" << (encounter.collision_risk ? "yes" : "no")
              << " DCPA=" << encounter.cpa.dcpa
              << " TCPA=" << encounter.cpa.tcpa
              << " alpha0_deg=" << rad2deg(encounter.initial_pose.contact_angle)
              << " beta0_deg=" << rad2deg(encounter.initial_pose.relative_bearing)
              << "\n";
    std::cout << "  rationale: " << encounter.rationale << "\n";
  }
}

void run_colreg_scan(
    const ScenarioSpec& scenario,
    const std::string& out_path,
    double dt,
    int scan_steps) {
  if (!scenario.own_ship.has_value() || scenario.target_ships.empty()) {
    throw std::runtime_error("COLREG scan requires own_ship and at least one target_ship.");
  }
  if (dt <= 0.0) {
    throw std::runtime_error("scan-dt must be > 0.");
  }
  if (scan_steps < 1) {
    throw std::runtime_error("scan-steps must be >= 1.");
  }

  OwnShip own = *scenario.own_ship;
  std::vector<TargetShip> targets = scenario.target_ships;
  std::vector<EncounterTrackState> track_states(targets.size());

  std::ofstream ofs(out_path);
  if (!ofs.is_open()) {
    throw std::runtime_error("Cannot open COLREG scan log file: " + out_path);
  }

  ofs << "step,time_s,target_id,own_x,own_y,own_course_deg,own_speed,"
      << "target_x,target_y,target_course_deg,target_speed,"
      << "type,role,geometry_type,geometry_role,risk,dcpa,tcpa,alpha0_deg,beta0_deg\n";

  for (int step = 0; step <= scan_steps; ++step) {
    const double time_s = static_cast<double>(step) * dt;

    for (int i = 0; i < static_cast<int>(targets.size()); ++i) {
      auto& target = targets[i];
      const ColregEncounter encounter = classify_encounter(own, target, scenario.colreg_cfg);
      const double range_now = current_range(own, target);
      const double clear_range = std::max(2.0 * scenario.colreg_cfg.risk_dcpa, scenario.colreg_cfg.risk_dcpa + 1.0);
      update_track_state(encounter, range_now, clear_range, &track_states[i]);

      const EncounterType output_type =
          track_states[i].engaged ? track_states[i].tracked_type : encounter.type;
      const ColregRole output_role =
          track_states[i].engaged ? track_states[i].tracked_role : encounter.own_role;
      ofs << step << ','
          << time_s << ','
          << target.id << ','
          << own.state.x << ','
          << own.state.y << ','
          << rad2deg(own.state.course) << ','
          << own.state.speed << ','
          << target.state.x << ','
          << target.state.y << ','
          << rad2deg(target.state.course) << ','
          << target.state.speed << ','
          << to_string(output_type) << ','
          << to_string(output_role) << ','
          << to_string(encounter.type) << ','
          << to_string(encounter.own_role) << ','
          << (encounter.collision_risk ? 1 : 0) << ','
          << encounter.cpa.dcpa << ','
          << encounter.cpa.tcpa << ','
          << rad2deg(encounter.initial_pose.contact_angle) << ','
          << rad2deg(encounter.initial_pose.relative_bearing) << '\n';
    }

    if (step == scan_steps) {
      continue;
    }

    own.state = propagate_const_velocity(own.state, dt);
    for (auto& target : targets) {
      target.state = propagate_const_velocity(target.state, dt);
    }
  }
}
}

int main(int argc, char** argv) {
  try {
    std::string scenario_path;
    std::string out_log = "receding_log.csv";
    std::string out_wp = "waypoints.csv";
    std::string out_colreg_log = "colreg_scan.csv";
    bool run_colreg_scan_mode = false;
    double scan_dt = 0.5;
    int scan_steps = 80;

    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help") {
        print_help(argv[0]);
        return 0;
      }
      if (arg == "--scenario" && i + 1 < argc) {
        scenario_path = argv[++i];
        continue;
      }
      if (arg == "--out-log" && i + 1 < argc) {
        out_log = argv[++i];
        continue;
      }
      if (arg == "--out-wp" && i + 1 < argc) {
        out_wp = argv[++i];
        continue;
      }
      if (arg == "--colreg-scan") {
        run_colreg_scan_mode = true;
        continue;
      }
      if (arg == "--out-colreg-log" && i + 1 < argc) {
        out_colreg_log = argv[++i];
        continue;
      }
      if (arg == "--scan-dt" && i + 1 < argc) {
        scan_dt = std::stod(argv[++i]);
        continue;
      }
      if (arg == "--scan-steps" && i + 1 < argc) {
        scan_steps = std::stoi(argv[++i]);
        continue;
      }
      throw std::runtime_error("Unknown or incomplete argument: " + arg);
    }

    ScenarioSpec scenario = scenario_path.empty() ? make_default_scenario() : load_scenario_ini(scenario_path);
    print_colreg_report(scenario);

    if (run_colreg_scan_mode) {
      run_colreg_scan(scenario, out_colreg_log, scan_dt, scan_steps);
      std::cout << "Saved COLREG scan log: " << out_colreg_log << "\n";
      std::cout << "Scenario source: " << scenario.source << "\n";
      return 0;
    }

    if (scenario.colreg_only) {
      std::cout << "Scenario source: " << scenario.source << "\n";
      return 0;
    }

    MPCNumericClothoidCost mpc(scenario.cfg);

    RecedingLog log = mpc.run_receding_horizon_multi(scenario.waypoints, scenario.initial_state, scenario.opts);
    mpc.write_log_csv(log, out_log);
    {
      std::ofstream ofs(out_wp);
      ofs << "idx,X,Y\n";
      for (int i = 0; i < static_cast<int>(scenario.waypoints.size()); ++i) {
        ofs << i << ',' << scenario.waypoints[i].X << ',' << scenario.waypoints[i].Y << '\n';
      }
    }

    std::cout << "Terminal [x,y,psi,K] = ["
              << log.traj.back().first << ", "
              << log.traj.back().second << ", "
              << log.psi.back() << ", "
              << log.K.back() << "]\n";
    std::cout << "Scenario source: " << scenario.source << "\n";
    std::cout << "Saved logs: " << out_log << ", " << out_wp << "\n";
    std::cout << "Active WP index: " << log.active_wp << "\n";
    std::cout << "Average solve time [s]: " << log.mean_solve_time_s << "\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }

  return 0;
}
