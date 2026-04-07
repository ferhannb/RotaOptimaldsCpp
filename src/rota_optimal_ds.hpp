#pragma once

#include <casadi/casadi.hpp>

#include <optional>
#include <string>
#include <utility>
#include <vector>

struct Waypoint {
  double X = 0.0;
  double Y = 0.0;
  std::optional<double> psig;
  std::optional<double> Kf;
  std::optional<double> tol;
  std::optional<double> w_wp;
  std::optional<double> hit_scale;
  bool use_Kf = true;
};

struct CircleObstacle {
  double cx = 0.0;
  double cy = 0.0;
  double radius = 0.0;
  bool enabled = true;
};

struct State4 {
  double x = 0.0;
  double y = 0.0;
  double psi = 0.0;
  double K = 0.0;
};

struct MPCConfig {
  int N = 25;
  double ds_min = 0.01;
  double ds_max = 1.0;
  double K_MAX = 0.30;
  double S_MAX = 14.0;
  int nseg = 4;

  double w_pos = 50.0;
  double w_psi = 250.0;
  double w_K = 0.5;
  double w_Kcmd = 0.5;
  double w_dKcmd = 2.0;
  double w_ds_smooth = 1.0;
  std::optional<double> ds_jump_max;
  double w_Kf = 10.0;
  bool enable_terminal_K_hard = false;

  int ipopt_max_iter = 2000;
  double ipopt_tol = 1e-6;

  std::vector<int> block_lengths_Kcmd;
  std::vector<int> block_lengths_ds;

  double w_prog = 0.0;
  double alpha_prog = 0.0;
  double hit_ratio = 0.7;
};

struct MPCSolution {
  std::vector<double> X;      // column-major, shape 4 x (N+1)
  std::vector<double> Kcmd;   // size N
  std::vector<double> ds;     // size N
  State4 start;
  State4 goal;
  double xwp = 0.0;
  double ywp = 0.0;
  double w_wp = 0.0;
  double term_scale = 1.0;
};

struct StepOutput {
  State4 state;
  double ds0 = 0.0;
  double Kcmd0 = 0.0;
  MPCSolution sol;
};

struct RecedingOptions {
  double tol_default = 2.5;
  int max_iters = 300;
  bool use_heading_gate = false;
  double tol_psi = 5.0 * 3.1415/ 180.0;
  double w_wp_intermediate = 0.0;
  double term_scale_intermediate = 0.1;
  double term_scale_final = 1.0;
  double hit_scale_intermediate = 1.0;
  double w_wp_final = 2.0;
  bool use_wp_kf = true;
  double kf_fallback = 0.0;

  bool enable_obstacle_avoidance = false;
  double obstacle_clearance = 0.5;
  double obstacle_trigger_margin = 0.5;
  double obstacle_waypoint_tol = 1.5;
  std::vector<CircleObstacle> obstacles;
};

struct RecedingLog {
  std::vector<std::pair<double, double>> traj;
  std::vector<double> psi;
  std::vector<double> K;
  std::vector<double> Kcmd;
  std::vector<double> ds;
  std::vector<double> detour_wp_x;
  std::vector<double> detour_wp_y;
  std::vector<double> detour_kf;
  std::vector<int> detour_obs_idx;
  std::vector<double> solve_time_s;
  double mean_solve_time_s = 0.0;
  State4 start;
  State4 goal;
  std::vector<Waypoint> waypoints;
  std::vector<int> wp_index;
  int active_wp = 0;
};

class MPCNumericClothoidCost {
 public:
  explicit MPCNumericClothoidCost(const MPCConfig& cfg);

  MPCSolution solve(
      double x0,
      double y0,
      double psi0,
      double K0,
      double xg,
      double yg,
      double psig,
      double Kf,
      double term_scale = 1.0,
      double w_wp = 0.0,
      std::optional<double> xwp = std::nullopt,
      std::optional<double> ywp = std::nullopt,
      double hit_scale = 0.0,
      std::optional<double> xhit = std::nullopt,
      std::optional<double> yhit = std::nullopt,
      std::optional<double> psihit = std::nullopt,
      std::optional<double> Khit = std::nullopt,
      std::optional<double> ds_prev = std::nullopt,
      std::optional<double> ds_seed = std::nullopt,
      bool use_last_warm = true);

  StepOutput mpc_step(
      const State4& state,
      const State4& goal,
      double term_scale = 1.0,
      double w_wp = 0.0,
      std::optional<double> xwp = std::nullopt,
      std::optional<double> ywp = std::nullopt,
      double hit_scale = 0.0,
      std::optional<double> xhit = std::nullopt,
      std::optional<double> yhit = std::nullopt,
      std::optional<double> psihit = std::nullopt,
      std::optional<double> Khit = std::nullopt,
      std::optional<double> ds_seed = std::nullopt);

  RecedingLog run_receding_horizon_multi(
      const std::vector<Waypoint>& waypoints,
      const State4& initial_state,
      const RecedingOptions& opts = RecedingOptions{});

  void write_log_csv(const RecedingLog& log, const std::string& path) const;

 private:
  struct WarmStartData {
    std::vector<double> X;
    std::vector<double> Kcmd;
    std::vector<double> ds;
    bool valid = false;
  };

  MPCConfig cfg_;
  int k_hit_ = 1;

  std::vector<int> bl_kcmd_;
  std::vector<int> bl_ds_;
  int NBK_ = 0;
  int NBd_ = 0;

  casadi::Opti opti_;
  casadi::MX X_;
  casadi::MX Kcmd_;
  casadi::MX ds_;
  std::optional<casadi::MX> KcmdB_;
  std::optional<casadi::MX> dsB_;

  casadi::MX x0_p_;
  casadi::MX y0_p_;
  casadi::MX psi0_p_;
  casadi::MX K0_p_;
  casadi::MX xg_p_;
  casadi::MX yg_p_;
  casadi::MX psig_p_;
  casadi::MX Kf_p_;
  casadi::MX xhit_p_;
  casadi::MX yhit_p_;
  casadi::MX psihit_p_;
  casadi::MX Khit_p_;
  casadi::MX hit_scale_p_;
  casadi::MX ds_prev_p_;
  casadi::MX term_scale_p_;
  casadi::MX w_wp_p_;
  casadi::MX xwp_p_;
  casadi::MX ywp_p_;

  std::optional<MPCSolution> last_sol_;
  WarmStartData last_warm_;
  double last_ds_applied_ = 0.0;
  double last_solve_time_s_ = 0.0;

  static casadi::MX wrap_to_pi(const casadi::MX& a);
  static casadi::MX sinc(const casadi::MX& x);

  static double wrap_to_pi_np(double a);
  static std::tuple<double, double, double> step_constK_sinc_np(
      double x,
      double y,
      double psi,
      double K,
      double ds);
  static std::tuple<double, double, double> clothoid_increment_numeric_np(
      double x0,
      double y0,
      double psi0,
      double K0,
      double K1,
      double ds,
      int nseg);
  static double K_next_fixed_ramp_np(
      double Kcur,
      double Kcmd,
      double ds,
      double K_MAX,
      double S_MAX,
      double eps = 1e-9);

  static std::tuple<casadi::MX, casadi::MX, casadi::MX> clothoid_increment_numeric(
      const casadi::MX& x0,
      const casadi::MX& y0,
      const casadi::MX& psi0,
      const casadi::MX& K0,
      const casadi::MX& K1,
      const casadi::MX& ds,
      int nseg);

  static casadi::MX K_next_fixed_ramp(
      const casadi::MX& Kcur,
      const casadi::MX& Kcmd,
      const casadi::MX& ds,
      double K_MAX,
      double S_MAX,
      double eps = 1e-9);

  void compute_block_maps();
  std::vector<double> block_init_from_full(
      const std::vector<double>& values,
      const std::vector<int>& blk_map,
      int n_blocks) const;

  void build_solver();
  void set_params(
      double x0,
      double y0,
      double psi0,
      double K0,
      double xg,
      double yg,
      double psig,
      double Kf,
      double term_scale,
      double w_wp,
      double xwp,
      double ywp,
      double xhit,
      double yhit,
      double psihit,
      double Khit,
      double hit_scale,
      double ds_prev);

  void warm_start(
      double x0,
      double y0,
      double psi0,
      double K0,
      double xg,
      double yg,
      std::optional<double> ds_seed);

  void apply_warm_start(
      double x0,
      double y0,
      double psi0,
      double K0,
      double xg,
      double yg,
      bool use_last_warm,
      std::optional<double> ds_seed);

  void shift_solution(const MPCSolution& sol);

  static int x_index(int row, int col) { return col * 4 + row; }
};
