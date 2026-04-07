#include "rota_optimal_ds.hpp"
#include "scenario_parser.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

namespace {
void print_help(const char* bin) {
  std::cout
      << "Usage: " << bin << " [options]\n"
      << "Options:\n"
      << "  --scenario <file>   Load scenario ini file\n" 
      << "  --out-log <file>    Output receding log csv (default: receding_log.csv)\n"
      << "  --out-wp <file>     Output waypoint csv (default: waypoints.csv)\n"
      << "  --help              Show this help\n";
}
}

int main(int argc, char** argv) {
  try {
    std::string scenario_path;
    std::string out_log = "receding_log.csv";
    std::string out_wp = "waypoints.csv";

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
      throw std::runtime_error("Unknown or incomplete argument: " + arg);
    }

    ScenarioSpec scenario = scenario_path.empty() ? make_default_scenario() : load_scenario_ini(scenario_path);

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
