#include "rota_optimal_ds.hpp"
#include "obstacle_avoidance.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <tuple>

using casadi::DM;
using casadi::MX;

namespace {
enum class NlpSolverKind {
  Ipopt,
  Fatrop,
};

double clip(double v, double lo, double hi) {
  return std::min(hi, std::max(lo, v));
}

std::string lower_copy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

NlpSolverKind resolve_primary_solver_kind(const MPCConfig& cfg) {
  const std::string requested = lower_copy(cfg.nlp_solver);
  if (requested == "fatrop") {
    return NlpSolverKind::Fatrop;
  }
  if (requested == "auto") {
    return cfg.realtime_mode ? NlpSolverKind::Fatrop : NlpSolverKind::Ipopt;
  }
  return NlpSolverKind::Ipopt;
}

bool fatrop_structured_supported(const MPCConfig& cfg) {
  const bool has_ds_jump = cfg.ds_jump_max.has_value() && (*cfg.ds_jump_max > 0.0);
  return cfg.block_lengths_Kcmd.empty() && cfg.block_lengths_ds.empty() && !has_ds_jump;
}

bool plugin_missing(const std::string& msg, const char* plugin) {
  const std::string quoted = std::string("Plugin '") + plugin + "' is not found";
  const std::string dylib = std::string("libcasadi_nlpsol_") + plugin;
  return (msg.find(quoted) != std::string::npos) || (msg.find(dylib) != std::string::npos);
}

std::string normalize_fatrop_structure_mode(std::string mode) {
  mode = lower_copy(std::move(mode));
  if (mode.empty() || mode == "none") {
    return "manual";
  }
  if (mode == "auto") {
    return "auto";
  }
  return "manual";
}

std::string normalize_fatrop_convexify_strategy(std::string strategy) {
  strategy = lower_copy(std::move(strategy));
  if (strategy == "regularize" || strategy == "eigen-reflect" || strategy == "eigen-clip") {
    return strategy;
  }
  return {};
}

double wp_tol(const Waypoint& wp, double tol_default) {
  return wp.tol.has_value() ? *wp.tol : tol_default;
}
}  // namespace

MPCNumericClothoidCost::MPCNumericClothoidCost(const MPCConfig& cfg) : cfg_(cfg) {
  if (cfg_.N < 2) {
    throw std::runtime_error("N must be >= 2");
  }
  k_hit_ = static_cast<int>(std::lround(cfg_.hit_ratio * static_cast<double>(cfg_.N)));
  if (k_hit_ < 1) {
    k_hit_ = 1;
  }
  if (k_hit_ > cfg_.N - 1) {
    k_hit_ = cfg_.N - 1;
  }

  compute_block_maps();
  build_solver();
  last_warm_.valid = false;
  last_ds_applied_ = cfg_.ds_max;
}

MX MPCNumericClothoidCost::wrap_to_pi(const MX& a) {
  return atan2(sin(a), cos(a));
}

MX MPCNumericClothoidCost::sinc(const MX& x) {
  return MX::if_else((x * x) < 1e-16, 1 - (x * x) / 6.0, sin(x) / x);
}

double MPCNumericClothoidCost::wrap_to_pi_np(double a) {
  return std::atan2(std::sin(a), std::cos(a));
}

std::tuple<double, double, double> MPCNumericClothoidCost::step_constK_sinc_np(
    double x,
    double y,
    double psi,
    double K,
    double ds) {
  const double dpsi = K * ds;
  const double half = dpsi / 2.0;
  const double fac = (std::abs(half) < 1e-12) ? (1.0 - (half * half) / 6.0) : (std::sin(half) / half);
  const double x1 = x + ds * fac * std::cos(psi + dpsi / 2.0);
  const double y1 = y + ds * fac * std::sin(psi + dpsi / 2.0);
  const double psi1 = psi + dpsi;
  return {x1, y1, psi1};
}

std::tuple<double, double, double> MPCNumericClothoidCost::clothoid_increment_numeric_np(
    double x0,
    double y0,
    double psi0,
    double K0,
    double K1,
    double ds,
    int nseg) {
  const double ds_seg = ds / static_cast<double>(nseg);
  double x = x0;
  double y = y0;
  double psi = psi0;
  for (int i = 0; i < nseg; ++i) {
    const double K_mid = K0 + (K1 - K0) * ((static_cast<double>(i) + 0.5) / static_cast<double>(nseg));
    std::tie(x, y, psi) = step_constK_sinc_np(x, y, psi, K_mid, ds_seg);
  }
  return {x, y, psi};
}

double MPCNumericClothoidCost::K_next_fixed_ramp_np(
    double Kcur,
    double Kcmd,
    double ds,
    double K_MAX,
    double S_MAX,
    double eps) {
  const double a0 = K_MAX / S_MAX;
  const double delta = Kcmd - Kcur;
  const double max_step = a0 * ds;
  const double dK = max_step * std::tanh(delta / (max_step + eps));
  return Kcur + dK;
}

std::tuple<MX, MX, MX> MPCNumericClothoidCost::clothoid_increment_numeric(
    const MX& x0,
    const MX& y0,
    const MX& psi0,
    const MX& K0,
    const MX& K1,
    const MX& ds,
    int nseg) {
  MX x = x0;
  MX y = y0;
  MX psi = psi0;
  const MX ds_seg = ds / static_cast<double>(nseg);

  for (int i = 0; i < nseg; ++i) {
    const MX K_mid = K0 + (K1 - K0) * ((static_cast<double>(i) + 0.5) / static_cast<double>(nseg));
    const MX dpsi = K_mid * ds_seg;
    const MX fac = sinc(dpsi / (2.0 * M_PI));
    const MX x1 = x + ds_seg * fac * cos(psi + dpsi / 2.0);
    const MX y1 = y + ds_seg * fac * sin(psi + dpsi / 2.0);
    const MX psi1 = psi + dpsi;
    x = x1;
    y = y1;
    psi = psi1;
  }
  return {x, y, psi};
}

MX MPCNumericClothoidCost::K_next_fixed_ramp(
    const MX& Kcur,
    const MX& Kcmd,
    const MX& ds,
    double K_MAX,
    double S_MAX,
    double eps) {
  const MX a0 = K_MAX / S_MAX;
  const MX delta = Kcmd - Kcur;
  const MX max_step = a0 * ds;
  const MX dK = max_step * tanh(delta / (max_step + eps));
  return Kcur + dK;
}

void MPCNumericClothoidCost::compute_block_maps() {
  auto make_map = [this](const std::vector<int>& lengths,
                         std::vector<int>* blk,
                         int* n_blocks) {
    if (lengths.empty()) {
      blk->clear();
      *n_blocks = 0;
      return;
    }

    const int sum = std::accumulate(lengths.begin(), lengths.end(), 0);
    if (sum != cfg_.N) {
      std::ostringstream oss;
      oss << "block_lengths sum must equal N (sum=" << sum << ", N=" << cfg_.N << ")";
      throw std::runtime_error(oss.str());
    }

    blk->assign(cfg_.N, 0);
    int s = 0;
    for (int i = 0; i < static_cast<int>(lengths.size()); ++i) {
      for (int j = 0; j < lengths[i]; ++j) {
        (*blk)[s + j] = i;
      }
      s += lengths[i];
    }
    *n_blocks = static_cast<int>(lengths.size());
  };

  make_map(cfg_.block_lengths_Kcmd, &bl_kcmd_, &NBK_);
  make_map(cfg_.block_lengths_ds, &bl_ds_, &NBd_);
}

void MPCNumericClothoidCost::build_solver() {
  const int N = cfg_.N;

  X_stage_.clear();
  Kcmd_stage_.clear();
  ds_stage_.clear();
  X_stage_.reserve(N + 1);
  Kcmd_stage_.reserve(N);
  ds_stage_.reserve(N);

  X_stage_.push_back(opti_.variable(4, 1));

  x0_p_ = opti_.parameter();
  y0_p_ = opti_.parameter();
  psi0_p_ = opti_.parameter();
  K0_p_ = opti_.parameter();

  xg_p_ = opti_.parameter();
  yg_p_ = opti_.parameter();
  psig_p_ = opti_.parameter();
  Kf_p_ = opti_.parameter();

  xhit_p_ = opti_.parameter();
  yhit_p_ = opti_.parameter();
  psihit_p_ = opti_.parameter();
  Khit_p_ = opti_.parameter();
  hit_scale_p_ = opti_.parameter();

  ds_prev_p_ = opti_.parameter();
  term_scale_p_ = opti_.parameter();
  w_wp_p_ = opti_.parameter();
  xwp_p_ = opti_.parameter();
  ywp_p_ = opti_.parameter();

  opti_.subject_to(X_stage_[0](0, 0) == x0_p_);
  opti_.subject_to(X_stage_[0](1, 0) == y0_p_);
  opti_.subject_to(X_stage_[0](2, 0) == psi0_p_);
  opti_.subject_to(X_stage_[0](3, 0) == K0_p_);
  opti_.subject_to(opti_.bounded(-cfg_.K_MAX, X_stage_[0](3, 0), cfg_.K_MAX));

  for (int k = 0; k < N; ++k) {
    const MX Kcmd_k = opti_.variable();
    const MX ds_k = opti_.variable();
    const MX x_next = opti_.variable(4, 1);

    Kcmd_stage_.push_back(Kcmd_k);
    ds_stage_.push_back(ds_k);
    X_stage_.push_back(x_next);

    opti_.subject_to(opti_.bounded(-cfg_.K_MAX, Kcmd_k, cfg_.K_MAX));
    opti_.subject_to(opti_.bounded(cfg_.ds_min, ds_k, cfg_.ds_max));
    opti_.subject_to(opti_.bounded(-cfg_.K_MAX, x_next(3, 0), cfg_.K_MAX));

    if (!bl_kcmd_.empty() && k > 0 && bl_kcmd_[k] == bl_kcmd_[k - 1]) {
      opti_.subject_to(Kcmd_k == Kcmd_stage_[k - 1]);
    }
    if (!bl_ds_.empty() && k > 0 && bl_ds_[k] == bl_ds_[k - 1]) {
      opti_.subject_to(ds_k == ds_stage_[k - 1]);
    }
  }

  X_ = MX::horzcat(X_stage_);
  Kcmd_ = MX::horzcat(Kcmd_stage_);
  ds_ = MX::horzcat(ds_stage_);

  if (cfg_.ds_jump_max.has_value() && *cfg_.ds_jump_max > 0.0) {
    const double jmax = std::abs(*cfg_.ds_jump_max);
    for (int k = 1; k < N; ++k) {
      opti_.subject_to(opti_.bounded(-jmax, ds_stage_[k] - ds_stage_[k - 1], jmax));
    }
    opti_.subject_to(opti_.bounded(-jmax, ds_stage_[0] - ds_prev_p_, jmax));
  }

  for (int k = 0; k < N; ++k) {
    const MX& xk = X_stage_[k];
    const MX& xk1 = X_stage_[k + 1];
    const MX& Kcmd_k = Kcmd_stage_[k];
    const MX& ds_k = ds_stage_[k];
    const MX K1 = K_next_fixed_ramp(xk(3, 0), Kcmd_k, ds_k, cfg_.K_MAX, cfg_.S_MAX);
    MX x1, y1, psi1;
    std::tie(x1, y1, psi1) =
        clothoid_increment_numeric(xk(0, 0), xk(1, 0), xk(2, 0), xk(3, 0), K1, ds_k, cfg_.nseg);
    opti_.subject_to(xk1(3, 0) == K1);
    opti_.subject_to(xk1(0, 0) == x1);
    opti_.subject_to(xk1(1, 0) == y1);
    opti_.subject_to(xk1(2, 0) == psi1);
  }

  MX obj = 0;
  for (int k = 0; k < N; ++k) {
    obj += cfg_.w_K * (X_stage_[k](3, 0) * X_stage_[k](3, 0));
    obj += cfg_.w_Kcmd * (Kcmd_stage_[k] * Kcmd_stage_[k]);
    if (k > 0) {
      obj += cfg_.w_dKcmd * ((Kcmd_stage_[k] - Kcmd_stage_[k - 1]) * (Kcmd_stage_[k] - Kcmd_stage_[k - 1]));
      obj += cfg_.w_ds_smooth * ((ds_stage_[k] - ds_stage_[k - 1]) * (ds_stage_[k] - ds_stage_[k - 1]));
    }
  }

  const MX Dref2 = (xg_p_ - x0_p_) * (xg_p_ - x0_p_) + (yg_p_ - y0_p_) * (yg_p_ - y0_p_);
  const MX pos_scale = Dref2 + 1.0;

  const MX pos_e = (X_stage_[N](0, 0) - xg_p_) * (X_stage_[N](0, 0) - xg_p_) +
                   (X_stage_[N](1, 0) - yg_p_) * (X_stage_[N](1, 0) - yg_p_);
  const MX psi_e = wrap_to_pi(X_stage_[N](2, 0) - psig_p_);
  const MX K_e = X_stage_[N](3, 0) - Kf_p_;
  obj += term_scale_p_ * (cfg_.w_pos * pos_e / pos_scale);
  obj += term_scale_p_ * (cfg_.w_psi * (psi_e * psi_e));
  obj += term_scale_p_ * (cfg_.w_Kf * (K_e * K_e));

  MX d2_sum = 0;
  for (int k = 1; k <= N; ++k) {
    d2_sum += (X_stage_[k](0, 0) - xwp_p_) * (X_stage_[k](0, 0) - xwp_p_) +
              (X_stage_[k](1, 0) - ywp_p_) * (X_stage_[k](1, 0) - ywp_p_);
  }
  obj += w_wp_p_ * (d2_sum / pos_scale);

  const int kh = k_hit_;
  const MX pos_hit = (X_stage_[kh](0, 0) - xhit_p_) * (X_stage_[kh](0, 0) - xhit_p_) +
                     (X_stage_[kh](1, 0) - yhit_p_) * (X_stage_[kh](1, 0) - yhit_p_);
  const MX psi_hit = wrap_to_pi(X_stage_[kh](2, 0) - psihit_p_);
  obj += hit_scale_p_ * (cfg_.w_pos * pos_hit / pos_scale);
  obj += hit_scale_p_ * (cfg_.w_psi * (psi_hit * psi_hit));
  obj += hit_scale_p_ * (cfg_.w_Kf * ((X_stage_[kh](3, 0) - Khit_p_) * (X_stage_[kh](3, 0) - Khit_p_)));

  if (cfg_.enable_terminal_K_hard) {
    opti_.subject_to(X_stage_[N](3, 0) == Kf_p_);
  }

  opti_.minimize(obj);
  configure_primary_solver();
}

void MPCNumericClothoidCost::configure_ipopt() {
  casadi::Dict p_opts;
  p_opts["print_time"] = true;

  casadi::Dict s_opts;
  s_opts["max_iter"] = cfg_.ipopt_max_iter;
  s_opts["print_level"] = 1;
  s_opts["tol"] = cfg_.ipopt_tol;
  s_opts["print_timing_statistics"] = "yes";
  opti_.solver("ipopt", p_opts, s_opts);
}

void MPCNumericClothoidCost::configure_fatrop() {
  casadi::Dict p_opts;
  p_opts["print_time"] = true;

  casadi::Dict s_opts;
  s_opts["max_iter"] = cfg_.ipopt_max_iter;
  s_opts["tol"] = cfg_.ipopt_tol;
  opti_.solver("fatrop", p_opts, s_opts);
}

void MPCNumericClothoidCost::configure_sqpmethod() {
  casadi::Dict sqp_opts;
  sqp_opts["print_time"] = true;
  sqp_opts["max_iter"] = cfg_.ipopt_max_iter;
  sqp_opts["print_header"] = false;
  sqp_opts["print_iteration"] = false;
  sqp_opts["tol_pr"] = cfg_.ipopt_tol;
  sqp_opts["tol_du"] = cfg_.ipopt_tol;
  sqp_opts["error_on_fail"] = false;
  sqp_opts["qpsol"] = "qrqp";
  casadi::Dict qpsol_opts;
  qpsol_opts["print_iter"] = false;
  qpsol_opts["print_header"] = false;
  qpsol_opts["error_on_fail"] = false;
  sqp_opts["qpsol_options"] = qpsol_opts;
  opti_.solver("sqpmethod", sqp_opts);
}

void MPCNumericClothoidCost::configure_primary_solver() {
  switch (resolve_primary_solver_kind(cfg_)) {
    case NlpSolverKind::Fatrop:
      if (fatrop_plugin_known_missing_ || fatrop_solver_failed_) {
        configure_ipopt();
        return;
      }
      configure_fatrop();
      return;
    case NlpSolverKind::Ipopt:
    default:
      configure_ipopt();
      return;
  }
}

void MPCNumericClothoidCost::build_fatrop_solver() {
  if (fatrop_solver_built_) {
    return;
  }

  const int N = cfg_.N;
  fatrop_x_offset_.assign(N + 1, -1);
  fatrop_u_offset_.assign(N, -1);
  fatrop_lbx_.clear();
  fatrop_ubx_.clear();
  fatrop_lbg_.clear();
  fatrop_ubg_.clear();

  std::vector<MX> x_stage(N + 1);
  std::vector<MX> u_stage(N);
  std::vector<MX> w;
  std::vector<MX> g;
  w.reserve(2 * N + 1);
  g.reserve(2 * N + 2);

  std::vector<int> nx(N + 1, 4);
  std::vector<int> nu(N + 1, 0);
  std::vector<int> ng(N + 1, 0);

  auto append_bounds = [this](const std::vector<double>& lb, const std::vector<double>& ub) {
    fatrop_lbx_.insert(fatrop_lbx_.end(), lb.begin(), lb.end());
    fatrop_ubx_.insert(fatrop_ubx_.end(), ub.begin(), ub.end());
  };
  auto append_eq = [this, &g, &ng](const MX& expr, int stage) {
    g.push_back(expr);
    for (casadi_int i = 0; i < expr.numel(); ++i) {
      fatrop_lbg_.push_back(0.0);
      fatrop_ubg_.push_back(0.0);
    }
    stage = std::max(0, std::min(stage, cfg_.N));
    ng[stage] += static_cast<int>(expr.numel());
  };
  auto append_rng = [this, &g, &ng](const MX& expr, double lb, double ub, int stage) {
    g.push_back(expr);
    for (casadi_int i = 0; i < expr.numel(); ++i) {
      fatrop_lbg_.push_back(lb);
      fatrop_ubg_.push_back(ub);
    }
    stage = std::max(0, std::min(stage, cfg_.N));
    ng[stage] += static_cast<int>(expr.numel());
  };

  MX p = MX::sym("p", 18);
  const MX x0_p = p(0);
  const MX y0_p = p(1);
  const MX psi0_p = p(2);
  const MX K0_p = p(3);
  const MX xg_p = p(4);
  const MX yg_p = p(5);
  const MX psig_p = p(6);
  const MX Kf_p = p(7);
  const MX xhit_p = p(8);
  const MX yhit_p = p(9);
  const MX psihit_p = p(10);
  const MX Khit_p = p(11);
  const MX hit_scale_p = p(12);
  const MX ds_prev_p = p(13);
  const MX term_scale_p = p(14);
  const MX w_wp_p = p(15);
  const MX xwp_p = p(16);
  const MX ywp_p = p(17);

  fatrop_x_offset_[0] = static_cast<int>(fatrop_lbx_.size());
  x_stage[0] = MX::sym("x_0", 4);
  w.push_back(x_stage[0]);
  append_bounds(
      {-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
       -std::numeric_limits<double>::infinity(), -cfg_.K_MAX},
      {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::infinity(), cfg_.K_MAX});

  for (int k = 0; k < N; ++k) {
    fatrop_u_offset_[k] = static_cast<int>(fatrop_lbx_.size());
    u_stage[k] = MX::sym("u_" + std::to_string(k), 2);
    w.push_back(u_stage[k]);
    nu[k] = 2;
    append_bounds({-cfg_.K_MAX, cfg_.ds_min}, {cfg_.K_MAX, cfg_.ds_max});

    fatrop_x_offset_[k + 1] = static_cast<int>(fatrop_lbx_.size());
    x_stage[k + 1] = MX::sym("x_" + std::to_string(k + 1), 4);
    w.push_back(x_stage[k + 1]);
    append_bounds(
        {-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
         -std::numeric_limits<double>::infinity(), -cfg_.K_MAX},
        {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity(), cfg_.K_MAX});

    const MX K1 = K_next_fixed_ramp(x_stage[k](3), u_stage[k](0), u_stage[k](1), cfg_.K_MAX, cfg_.S_MAX);
    MX x1, y1, psi1;
    std::tie(x1, y1, psi1) =
        clothoid_increment_numeric(x_stage[k](0), x_stage[k](1), x_stage[k](2), x_stage[k](3), K1, u_stage[k](1), cfg_.nseg);
    g.push_back(x_stage[k + 1] - MX::vertcat({x1, y1, psi1, K1}));
    for (int i = 0; i < 4; ++i) {
      fatrop_lbg_.push_back(0.0);
      fatrop_ubg_.push_back(0.0);
    }
    if (k == 0) {
      append_eq(x_stage[0] - MX::vertcat({x0_p, y0_p, psi0_p, K0_p}), 0);
    }
  }

  MX obj = 0;
  for (int k = 0; k < N; ++k) {
    obj += cfg_.w_K * (x_stage[k](3) * x_stage[k](3));
    obj += cfg_.w_Kcmd * (u_stage[k](0) * u_stage[k](0));
    if (k > 0) {
      obj += cfg_.w_dKcmd * ((u_stage[k](0) - u_stage[k - 1](0)) * (u_stage[k](0) - u_stage[k - 1](0)));
      obj += cfg_.w_ds_smooth * ((u_stage[k](1) - u_stage[k - 1](1)) * (u_stage[k](1) - u_stage[k - 1](1)));
    }
  }

  const MX Dref2 = (xg_p - x0_p) * (xg_p - x0_p) + (yg_p - y0_p) * (yg_p - y0_p);
  const MX pos_scale = Dref2 + 1.0;
  const MX pos_e = (x_stage[N](0) - xg_p) * (x_stage[N](0) - xg_p) +
                   (x_stage[N](1) - yg_p) * (x_stage[N](1) - yg_p);
  const MX psi_e = wrap_to_pi(x_stage[N](2) - psig_p);
  const MX K_e = x_stage[N](3) - Kf_p;
  obj += term_scale_p * (cfg_.w_pos * pos_e / pos_scale);
  obj += term_scale_p * (cfg_.w_psi * (psi_e * psi_e));
  obj += term_scale_p * (cfg_.w_Kf * (K_e * K_e));

  MX d2_sum = 0;
  for (int k = 1; k <= N; ++k) {
    d2_sum += (x_stage[k](0) - xwp_p) * (x_stage[k](0) - xwp_p) +
              (x_stage[k](1) - ywp_p) * (x_stage[k](1) - ywp_p);
  }
  obj += w_wp_p * (d2_sum / pos_scale);

  const int kh = k_hit_;
  const MX pos_hit = (x_stage[kh](0) - xhit_p) * (x_stage[kh](0) - xhit_p) +
                     (x_stage[kh](1) - yhit_p) * (x_stage[kh](1) - yhit_p);
  const MX psi_hit = wrap_to_pi(x_stage[kh](2) - psihit_p);
  obj += hit_scale_p * (cfg_.w_pos * pos_hit / pos_scale);
  obj += hit_scale_p * (cfg_.w_psi * (psi_hit * psi_hit));
  obj += hit_scale_p * (cfg_.w_Kf * ((x_stage[kh](3) - Khit_p) * (x_stage[kh](3) - Khit_p)));

  if (cfg_.enable_terminal_K_hard) {
    append_eq(x_stage[N](3) - Kf_p, N);
  }

  casadi::MXDict nlp;
  nlp["x"] = MX::vertcat(w);
  nlp["f"] = obj;
  nlp["g"] = MX::vertcat(g);
  nlp["p"] = p;

  casadi::Dict opts;
  opts["print_time"] = true;

  const std::string structure_mode = normalize_fatrop_structure_mode(cfg_.fatrop_structure_detection);
  opts["structure_detection"] = structure_mode;
  if (structure_mode == "manual") {
    opts["N"] = N;
    opts["nx"] = nx;
    opts["nu"] = nu;
    opts["ng"] = ng;
  }
  const std::string convexify = normalize_fatrop_convexify_strategy(cfg_.fatrop_convexify_strategy);
  if (!convexify.empty()) {
    opts["convexify_strategy"] = convexify;
    opts["convexify_margin"] = cfg_.fatrop_convexify_margin;
  }
  opts["debug"] = cfg_.fatrop_debug;

  fatrop_solver_ = casadi::nlpsol("fatrop_mpc", "fatrop", nlp, opts);
  fatrop_solver_built_ = true;
}

void MPCNumericClothoidCost::set_params(
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
    double ds_prev) {
  opti_.set_value(x0_p_, x0);
  opti_.set_value(y0_p_, y0);
  opti_.set_value(psi0_p_, psi0);
  opti_.set_value(K0_p_, K0);

  opti_.set_value(xg_p_, xg);
  opti_.set_value(yg_p_, yg);
  opti_.set_value(psig_p_, psig);
  opti_.set_value(Kf_p_, Kf);

  opti_.set_value(xhit_p_, xhit);
  opti_.set_value(yhit_p_, yhit);
  opti_.set_value(psihit_p_, psihit);
  opti_.set_value(Khit_p_, Khit);
  opti_.set_value(hit_scale_p_, hit_scale);

  opti_.set_value(ds_prev_p_, ds_prev);
  opti_.set_value(term_scale_p_, term_scale);
  opti_.set_value(w_wp_p_, w_wp);
  opti_.set_value(xwp_p_, xwp);
  opti_.set_value(ywp_p_, ywp);
}

void MPCNumericClothoidCost::set_opti_initial_from_warm(const WarmStartData& ws) {
  for (int k = 0; k < cfg_.N; ++k) {
    opti_.set_initial(Kcmd_stage_[k], ws.Kcmd[k]);
    opti_.set_initial(ds_stage_[k], ws.ds[k]);
  }
  for (int k = 0; k <= cfg_.N; ++k) {
    opti_.set_initial(X_stage_[k](0, 0), ws.X[x_index(0, k)]);
    opti_.set_initial(X_stage_[k](1, 0), ws.X[x_index(1, k)]);
    opti_.set_initial(X_stage_[k](2, 0), ws.X[x_index(2, k)]);
    opti_.set_initial(X_stage_[k](3, 0), ws.X[x_index(3, k)]);
  }
}

MPCNumericClothoidCost::WarmStartData MPCNumericClothoidCost::make_warm_start_data(
    double x0,
    double y0,
    double psi0,
    double K0,
    double xg,
    double yg,
    double psig,
    bool use_last_warm,
    std::optional<double> ds_seed) const {
  if (use_last_warm && last_warm_.valid) {
    return last_warm_;
  }

  WarmStartData ws;
  const double dist = std::hypot(xg - x0, yg - y0);
  const double dpsi_target = wrap_to_pi_np(psig - psi0);
  const bool heading_only_turn = (dist < 1e-3) && (std::abs(dpsi_target) > 0.25);
  const double psi_goal = heading_only_turn ? psig : std::atan2(yg - y0, xg - x0);
  const double dpsi_goal = heading_only_turn ? dpsi_target : wrap_to_pi_np(psi_goal - psi0);
  const double heading_span = std::max(std::abs(dpsi_target), 1e-3);
  const double kcmd_den = heading_only_turn ? (0.5 * static_cast<double>(cfg_.N) * cfg_.ds_max) : std::max(dist, 1e-3);
  const double Kcmd_guess = clip(dpsi_goal / std::max(kcmd_den, 1e-3), -cfg_.K_MAX, cfg_.K_MAX);

  ws.Kcmd.assign(cfg_.N, Kcmd_guess);

  double ds_guess = 0.0;
  if (ds_seed.has_value()) {
    ds_guess = clip(*ds_seed, cfg_.ds_min, cfg_.ds_max);
  } else {
    if (heading_only_turn) {
      const double ds_turn = heading_span / (0.75 * static_cast<double>(cfg_.N) * std::max(cfg_.K_MAX, 1e-6));
      ds_guess = clip(ds_turn, cfg_.ds_min, cfg_.ds_max);
    } else {
      ds_guess = clip(dist / std::max(cfg_.N, 1), cfg_.ds_min, cfg_.ds_max);
    }
    if (dist > 2.0) {
      ds_guess = std::max(ds_guess, 0.6 * cfg_.ds_max);
    }
  }
  ws.ds.assign(cfg_.N, ds_guess);

  std::vector<double> x_ws(cfg_.N + 1, 0.0);
  std::vector<double> y_ws(cfg_.N + 1, 0.0);
  std::vector<double> psi_ws(cfg_.N + 1, 0.0);
  std::vector<double> K_ws(cfg_.N + 1, 0.0);
  x_ws[0] = x0;
  y_ws[0] = y0;
  psi_ws[0] = psi0;
  K_ws[0] = K0;

  for (int k = 0; k < cfg_.N; ++k) {
    K_ws[k + 1] = K_next_fixed_ramp_np(K_ws[k], ws.Kcmd[k], ws.ds[k], cfg_.K_MAX, cfg_.S_MAX);
    std::tie(x_ws[k + 1], y_ws[k + 1], psi_ws[k + 1]) =
        clothoid_increment_numeric_np(x_ws[k], y_ws[k], psi_ws[k], K_ws[k], K_ws[k + 1], ws.ds[k], cfg_.nseg);
  }

  ws.X.assign(4 * (cfg_.N + 1), 0.0);
  for (int k = 0; k <= cfg_.N; ++k) {
    ws.X[x_index(0, k)] = x_ws[k];
    ws.X[x_index(1, k)] = y_ws[k];
    ws.X[x_index(2, k)] = psi_ws[k];
    ws.X[x_index(3, k)] = K_ws[k];
  }
  ws.valid = true;
  return ws;
}

std::vector<double> MPCNumericClothoidCost::make_param_vector(
    double x0,
    double y0,
    double psi0,
    double K0,
    double xg,
    double yg,
    double psig,
    double Kf,
    double xhit,
    double yhit,
    double psihit,
    double Khit,
    double hit_scale,
    double ds_prev,
    double term_scale,
    double w_wp,
    double xwp,
    double ywp) const {
  return {x0, y0, psi0, K0, xg, yg, psig, Kf, xhit, yhit, psihit, Khit, hit_scale, ds_prev, term_scale, w_wp, xwp, ywp};
}

std::optional<MPCSolution> MPCNumericClothoidCost::solve_with_fatrop(
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
    double ds_prev,
    const WarmStartData& warm,
    std::string* msg) {
  try {
    build_fatrop_solver();

    std::vector<double> x0_vec(fatrop_lbx_.size(), 0.0);
    for (int k = 0; k <= cfg_.N; ++k) {
      const int off = fatrop_x_offset_[k];
      x0_vec[off + 0] = warm.X[x_index(0, k)];
      x0_vec[off + 1] = warm.X[x_index(1, k)];
      x0_vec[off + 2] = warm.X[x_index(2, k)];
      x0_vec[off + 3] = warm.X[x_index(3, k)];
      if (k < cfg_.N) {
        const int uoff = fatrop_u_offset_[k];
        x0_vec[uoff + 0] = warm.Kcmd[k];
        x0_vec[uoff + 1] = warm.ds[k];
      }
    }

    casadi::DMDict arg;
    arg["x0"] = DM(x0_vec);
    arg["p"] = DM(make_param_vector(
        x0, y0, psi0, K0, xg, yg, psig, Kf, xhit, yhit, psihit, Khit, hit_scale, ds_prev, term_scale, w_wp, xwp, ywp));
    arg["lbx"] = DM(fatrop_lbx_);
    arg["ubx"] = DM(fatrop_ubx_);
    arg["lbg"] = DM(fatrop_lbg_);
    arg["ubg"] = DM(fatrop_ubg_);

    const casadi::DMDict res = fatrop_solver_(arg);
    const std::vector<double> x_opt = res.at("x").nonzeros();

    MPCSolution out;
    out.X.assign(4 * (cfg_.N + 1), 0.0);
    out.Kcmd.assign(cfg_.N, 0.0);
    out.ds.assign(cfg_.N, 0.0);
    for (int k = 0; k <= cfg_.N; ++k) {
      const int off = fatrop_x_offset_[k];
      out.X[x_index(0, k)] = x_opt[off + 0];
      out.X[x_index(1, k)] = x_opt[off + 1];
      out.X[x_index(2, k)] = x_opt[off + 2];
      out.X[x_index(3, k)] = x_opt[off + 3];
      if (k < cfg_.N) {
        const int uoff = fatrop_u_offset_[k];
        out.Kcmd[k] = x_opt[uoff + 0];
        out.ds[k] = x_opt[uoff + 1];
      }
    }
    out.start = State4{x0, y0, psi0, K0};
    out.goal = State4{xg, yg, psig, Kf};
    out.xwp = xwp;
    out.ywp = ywp;
    out.w_wp = w_wp;
    out.term_scale = term_scale;
    return out;
  } catch (const std::exception& e) {
    if (msg) {
      *msg = e.what();
    }
    return std::nullopt;
  }
}

void MPCNumericClothoidCost::warm_start(
    double x0,
    double y0,
    double psi0,
    double K0,
    double xg,
    double yg,
    double psig,
    std::optional<double> ds_seed) {
  set_opti_initial_from_warm(make_warm_start_data(x0, y0, psi0, K0, xg, yg, psig, false, ds_seed));
}

void MPCNumericClothoidCost::apply_warm_start(
    double x0,
    double y0,
    double psi0,
    double K0,
    double xg,
    double yg,
    double psig,
    bool use_last_warm,
    std::optional<double> ds_seed) {
  set_opti_initial_from_warm(make_warm_start_data(x0, y0, psi0, K0, xg, yg, psig, use_last_warm, ds_seed));
}

void MPCNumericClothoidCost::shift_solution(const MPCSolution& sol) {
  WarmStartData ws;
  ws.X.resize(sol.X.size());
  ws.Kcmd.resize(sol.Kcmd.size());
  ws.ds.resize(sol.ds.size());

  for (int k = 0; k < cfg_.N; ++k) {
    ws.X[x_index(0, k)] = sol.X[x_index(0, k + 1)];
    ws.X[x_index(1, k)] = sol.X[x_index(1, k + 1)];
    ws.X[x_index(2, k)] = sol.X[x_index(2, k + 1)];
    ws.X[x_index(3, k)] = sol.X[x_index(3, k + 1)];
  }
  ws.X[x_index(0, cfg_.N)] = sol.X[x_index(0, cfg_.N)];
  ws.X[x_index(1, cfg_.N)] = sol.X[x_index(1, cfg_.N)];
  ws.X[x_index(2, cfg_.N)] = sol.X[x_index(2, cfg_.N)];
  ws.X[x_index(3, cfg_.N)] = sol.X[x_index(3, cfg_.N)];

  for (int k = 0; k < cfg_.N - 1; ++k) {
    ws.Kcmd[k] = sol.Kcmd[k + 1];
    ws.ds[k] = sol.ds[k + 1];
  }
  ws.Kcmd[cfg_.N - 1] = sol.Kcmd[cfg_.N - 1];
  ws.ds[cfg_.N - 1] = sol.ds[cfg_.N - 1];

  ws.valid = true;
  last_warm_ = ws;
}

MPCSolution MPCNumericClothoidCost::solve(
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
    std::optional<double> xwp,
    std::optional<double> ywp,
    double hit_scale,
    std::optional<double> xhit,
    std::optional<double> yhit,
    std::optional<double> psihit,
    std::optional<double> Khit,
    std::optional<double> ds_prev,
    std::optional<double> ds_seed,
    bool use_last_warm) {
  const double xwp_v = xwp.has_value() ? *xwp : xg;
  const double ywp_v = ywp.has_value() ? *ywp : yg;
  const double xhit_v = xhit.has_value() ? *xhit : xg;
  const double yhit_v = yhit.has_value() ? *yhit : yg;
  const double psihit_v = psihit.has_value() ? *psihit : psig;
  const double Khit_v = Khit.has_value() ? *Khit : Kf;
  const double ds_prev_v = ds_prev.has_value() ? *ds_prev : last_ds_applied_;
  const NlpSolverKind primary_solver = resolve_primary_solver_kind(cfg_);
  const WarmStartData warm = make_warm_start_data(x0, y0, psi0, K0, xg, yg, psig, use_last_warm, ds_seed);

  auto t0 = std::chrono::steady_clock::now();
  std::string msg;
  std::optional<MPCSolution> sol;

  bool using_ipopt_sequence = (primary_solver == NlpSolverKind::Ipopt);
  if (primary_solver == NlpSolverKind::Fatrop &&
      !(fatrop_plugin_known_missing_ || fatrop_solver_failed_)) {
    const bool try_structured = fatrop_structured_supported(cfg_);
    if (try_structured) {
      sol = solve_with_fatrop(
          x0,
          y0,
          psi0,
          K0,
          xg,
          yg,
          psig,
          Kf,
          term_scale,
          w_wp,
          xwp_v,
          ywp_v,
          xhit_v,
          yhit_v,
          psihit_v,
          Khit_v,
          hit_scale,
          ds_prev_v,
          warm,
          &msg);
      if (!sol.has_value()) {
        std::cerr << "[WARN] Structured FATROP solve failed, trying generic FATROP.\n";
        std::cerr << "[WARN] FATROP error: " << msg << '\n';
      }
    }

    if (!sol.has_value()) {
      set_params(
          x0,
          y0,
          psi0,
          K0,
          xg,
          yg,
          psig,
          Kf,
          term_scale,
          w_wp,
          xwp_v,
          ywp_v,
          xhit_v,
          yhit_v,
          psihit_v,
          Khit_v,
          hit_scale,
          ds_prev_v);
      configure_fatrop();
      set_opti_initial_from_warm(warm);
      try {
        casadi::OptiSol opti_fatrop = opti_.solve();
        MPCSolution opti_out;
        opti_out.X = opti_fatrop.value(X_).nonzeros();
        opti_out.Kcmd = opti_fatrop.value(Kcmd_).nonzeros();
        opti_out.ds = opti_fatrop.value(ds_).nonzeros();
        opti_out.start = State4{x0, y0, psi0, K0};
        opti_out.goal = State4{xg, yg, psig, Kf};
        opti_out.xwp = xwp_v;
        opti_out.ywp = ywp_v;
        opti_out.w_wp = w_wp;
        opti_out.term_scale = term_scale;
        sol = std::move(opti_out);
      } catch (const std::exception& e) {
        msg = e.what();
        if (plugin_missing(msg, "fatrop")) {
          fatrop_plugin_known_missing_ = true;
          std::cerr << "[WARN] FATROP not available, falling back to IPOPT.\n";
        } else {
          fatrop_solver_failed_ = true;
          std::cerr << "[WARN] FATROP solve failed, falling back to IPOPT.\n";
          std::cerr << "[WARN] FATROP error: " << msg << '\n';
        }
      }
    }

    if (sol.has_value()) {
      using_ipopt_sequence = false;
    } else {
      using_ipopt_sequence = true;
    }
  } else if (primary_solver == NlpSolverKind::Fatrop) {
    using_ipopt_sequence = true;
  }

  if (!sol.has_value() && using_ipopt_sequence) {
    set_params(
        x0,
        y0,
        psi0,
        K0,
        xg,
        yg,
        psig,
        Kf,
        term_scale,
        w_wp,
        xwp_v,
        ywp_v,
        xhit_v,
        yhit_v,
        psihit_v,
        Khit_v,
        hit_scale,
        ds_prev_v);
    configure_ipopt();
    set_opti_initial_from_warm(warm);

    const bool ipopt_missing = plugin_missing(msg, "ipopt");
    const bool restoration_failed = msg.find("Restoration_Failed") != std::string::npos;
    std::optional<casadi::OptiSol> opti_sol;
    auto try_current_solver = [this, &opti_sol, &msg]() -> bool {
      try {
        opti_sol = opti_.solve();
        return true;
      } catch (const std::exception& e) {
        msg = e.what();
        opti_sol.reset();
        return false;
      }
    };

    try_current_solver();

    if ((!ipopt_missing) && use_last_warm && !opti_sol.has_value()) {
      std::cerr << "[WARN] IPOPT solve failed, retrying with fresh warm start.\n";
      set_opti_initial_from_warm(make_warm_start_data(x0, y0, psi0, K0, xg, yg, psig, false, ds_seed));
      if (!try_current_solver()) {
        const std::string retry_msg = msg;
        if ((!restoration_failed) &&
            (retry_msg.find("Restoration_Failed") == std::string::npos) &&
            (!ipopt_missing)) {
          throw std::runtime_error(retry_msg);
        }
        std::cerr << "[WARN] Fresh IPOPT retry failed, falling back to sqpmethod.\n";
      }
    }

    if (!opti_sol.has_value()) {
      if (ipopt_missing) {
        std::cerr << "[WARN] IPOPT not available, falling back to sqpmethod.\n";
      } else if (restoration_failed || use_last_warm || primary_solver == NlpSolverKind::Fatrop) {
        std::cerr << "[WARN] Switching to sqpmethod fallback for this step.\n";
      } else {
        throw std::runtime_error(msg);
      }

      configure_sqpmethod();
      set_opti_initial_from_warm(make_warm_start_data(x0, y0, psi0, K0, xg, yg, psig, false, ds_seed));
      opti_sol = opti_.solve();
      configure_primary_solver();
    }

    if (opti_sol.has_value()) {
      MPCSolution opti_out;
      opti_out.X = opti_sol->value(X_).nonzeros();
      opti_out.Kcmd = opti_sol->value(Kcmd_).nonzeros();
      opti_out.ds = opti_sol->value(ds_).nonzeros();
      opti_out.start = State4{x0, y0, psi0, K0};
      opti_out.goal = State4{xg, yg, psig, Kf};
      opti_out.xwp = xwp_v;
      opti_out.ywp = ywp_v;
      opti_out.w_wp = w_wp;
      opti_out.term_scale = term_scale;
      sol = std::move(opti_out);
    }
  }

  if (!sol.has_value()) {
    throw std::runtime_error(msg);
  }
  auto t1 = std::chrono::steady_clock::now();
  const std::chrono::duration<double> dt = t1 - t0;
  last_solve_time_s_ = dt.count();

  std::cout << "CasADi solve time: " << std::fixed << std::setprecision(3) << dt.count() << " s\n";

  last_sol_ = *sol;
  return *sol;
}

StepOutput MPCNumericClothoidCost::mpc_step(
    const State4& state,
    const State4& goal,
    double term_scale,
    double w_wp,
    std::optional<double> xwp,
    std::optional<double> ywp,
    double hit_scale,
    std::optional<double> xhit,
    std::optional<double> yhit,
    std::optional<double> psihit,
    std::optional<double> Khit,
    std::optional<double> ds_seed) {
  auto sol = solve(
      state.x,
      state.y,
      state.psi,
      state.K,
      goal.x,
      goal.y,
      goal.psi,
      goal.K,
      term_scale,
      w_wp,
      xwp,
      ywp,
      hit_scale,
      xhit,
      yhit,
      psihit,
      Khit,
      last_ds_applied_,
      ds_seed,
      true);

  const double ds0 = sol.ds.front();
  const double K1 = sol.X[x_index(3, 1)];

  double x1, y1, psi1;
  std::tie(x1, y1, psi1) =
      clothoid_increment_numeric_np(state.x, state.y, state.psi, state.K, K1, ds0, cfg_.nseg);
  last_ds_applied_ = ds0;

  shift_solution(sol);

  StepOutput out;
  out.state = State4{x1, y1, psi1, K1};
  out.ds0 = ds0;
  out.Kcmd0 = sol.Kcmd.front();
  out.sol = std::move(sol);
  return out;
}

RecedingLog MPCNumericClothoidCost::run_receding_horizon_multi(
    const std::vector<Waypoint>& waypoints,
    const State4& initial_state,
    const RecedingOptions& opts) {
  if (waypoints.empty()) {
    throw std::runtime_error("Waypoint list cannot be empty.");
  }

  auto pick_heading = [](double x, double y, const Waypoint& wp) {
    if (!wp.psig.has_value()) {
      return std::atan2(wp.Y - y, wp.X - x);
    }
    return *wp.psig;
  };

  auto pick_Kf = [&opts](const Waypoint& wp) {
    if ((!opts.use_wp_kf) || (!wp.use_Kf)) {
      return opts.kf_fallback;
    }
    if (!wp.Kf.has_value()) {
      return opts.kf_fallback;
    }
    return *wp.Kf;
  };

  last_warm_.valid = false;
  last_ds_applied_ = cfg_.ds_max;

  State4 state = initial_state;
  double x = state.x;
  double y = state.y;
  double psi = state.psi;
  double K = state.K;

  RecedingLog log;
  log.traj.push_back({x, y});
  log.psi.push_back(psi);
  log.K.push_back(K);
  log.detour_wp_x.push_back(std::numeric_limits<double>::quiet_NaN());
  log.detour_wp_y.push_back(std::numeric_limits<double>::quiet_NaN());
  log.detour_kf.push_back(std::numeric_limits<double>::quiet_NaN());
  log.detour_obs_idx.push_back(-1);
  log.start = initial_state;
  log.waypoints = waypoints;
  log.wp_index.push_back(0);

  std::optional<double> ds_seed_next = std::nullopt;
  int cur_idx = 0;
  std::vector<bool> obstacle_done_for_wp(opts.obstacles.size(), false);
  std::optional<Waypoint> detour_wp = std::nullopt;
  int detour_obstacle_idx = -1;

  for (int it = 0; it < opts.max_iters; ++it) {
    const Waypoint& wp_main = waypoints[cur_idx];

    if ((!detour_wp.has_value()) && opts.enable_obstacle_avoidance && !opts.obstacles.empty()) {
      auto detour = select_obstacle_detour_waypoint(
          x,
          y,
          psi,
          wp_main,
          opts.obstacles,
          obstacle_done_for_wp,
          opts.obstacle_clearance,
          opts.obstacle_trigger_margin,
          opts.obstacle_waypoint_tol);
      if (detour.has_value()) {
        detour_wp = detour->waypoint;
        detour_obstacle_idx = detour->obstacle_index;
        last_warm_.valid = false;
        ds_seed_next = log.ds.empty() ? std::optional<double>{} : std::optional<double>{log.ds.back()};
        const double detour_kf = detour_wp->Kf.has_value() ? *detour_wp->Kf : opts.kf_fallback;
        std::cout << "[INFO] Obstacle detour activated (obs=" << detour_obstacle_idx
                  << ") via [" << detour_wp->X << ", " << detour_wp->Y << "], Kf=" << detour_kf << '\n';
      }
    }

    const bool using_detour = detour_wp.has_value();
    const Waypoint& wp = using_detour ? *detour_wp : wp_main;
    const double Xf = wp.X;
    const double Yf = wp.Y;
    const double psig = pick_heading(x, y, wp);
    const double Kf = pick_Kf(wp);

    const double tol_here = wp_tol(wp, opts.tol_default);
    const bool is_last = (!using_detour) && (cur_idx >= static_cast<int>(waypoints.size()) - 1);

    const double dist_now = std::hypot(x - Xf, y - Yf);
    bool heading_ok = true;
    if (opts.use_heading_gate && !using_detour && wp.psig.has_value()) {
      heading_ok = std::abs(wrap_to_pi_np(psi - psig)) < opts.tol_psi;
    }

    if (using_detour && (dist_now <= tol_here)) {
      if (detour_obstacle_idx >= 0 && detour_obstacle_idx < static_cast<int>(obstacle_done_for_wp.size())) {
        obstacle_done_for_wp[detour_obstacle_idx] = true;
      }
      detour_wp = std::nullopt;
      detour_obstacle_idx = -1;
      last_warm_.valid = false;
      ds_seed_next = log.ds.empty() ? std::optional<double>{} : std::optional<double>{log.ds.back()};
      continue;
    }

    if ((!using_detour) && (!is_last) && (dist_now <= tol_here) && heading_ok) {
      last_warm_.valid = false;
      ds_seed_next = log.ds.empty() ? std::optional<double>{} : std::optional<double>{log.ds.back()};
      cur_idx += 1;
      std::fill(obstacle_done_for_wp.begin(), obstacle_done_for_wp.end(), false);
      continue;
    }

    double term_scale = 0.0;
    double w_wp = 0.0;
    double hit_scale = 0.0;
    double xwp = Xf;
    double ywp = Yf;
    double xhit = Xf;
    double yhit = Yf;
    double psihit = psig;
    double Khit = Kf;
    if (is_last) {
      const double goal_from_start = std::hypot(wp_main.X - initial_state.x, wp_main.Y - initial_state.y);
      const double heading_from_start = std::abs(wrap_to_pi_np(psig - initial_state.psi));
      const bool same_point_heading_turn = (goal_from_start <= tol_here) && (heading_from_start > opts.tol_psi);

      term_scale = opts.term_scale_final;
      w_wp = same_point_heading_turn ? 0.0 : opts.w_wp_final;
      hit_scale = 0.0;

      if (same_point_heading_turn) {
        const double dpsi_total = wrap_to_pi_np(psig - initial_state.psi);
        const double psi_mid = wrap_to_pi_np(initial_state.psi + 0.5 * dpsi_total);
        const double loop_radius = std::max(2.0, 1.0 / std::max(cfg_.K_MAX, 1e-6));
        xhit = Xf + loop_radius * std::cos(psi_mid);
        yhit = Yf + loop_radius * std::sin(psi_mid);
        psihit = psi_mid;
        xwp = xhit;
        ywp = yhit;
        w_wp = std::max(0.5, opts.w_wp_final);
        hit_scale = 0.75 * opts.term_scale_final;
      }
    } else {
      term_scale = opts.term_scale_intermediate;
      w_wp = wp.w_wp.has_value() ? *wp.w_wp : opts.w_wp_intermediate;
      hit_scale = wp.hit_scale.has_value() ? *wp.hit_scale : opts.hit_scale_intermediate;
    }

    StepOutput step;
    try {
      step = mpc_step(
          state,
          State4{Xf, Yf, psig, Kf},
          term_scale,
          w_wp,
          xwp,
          ywp,
          hit_scale,
          xhit,
          yhit,
          psihit,
          Khit,
          ds_seed_next);
    } catch (const std::exception& e) {
      std::cerr << "[WARN] MPC step failed at iter " << it << ": " << e.what() << '\n';
      break;
    }

    ds_seed_next = std::nullopt;

    state = step.state;
    x = state.x;
    y = state.y;
    psi = state.psi;
    K = state.K;

    log.traj.push_back({x, y});
    log.psi.push_back(psi);
    log.K.push_back(K);
    log.Kcmd.push_back(step.Kcmd0);
    log.ds.push_back(step.ds0);
    if (using_detour) {
      log.detour_wp_x.push_back(Xf);
      log.detour_wp_y.push_back(Yf);
      log.detour_kf.push_back(Kf);
      log.detour_obs_idx.push_back(detour_obstacle_idx);
    } else {
      log.detour_wp_x.push_back(std::numeric_limits<double>::quiet_NaN());
      log.detour_wp_y.push_back(std::numeric_limits<double>::quiet_NaN());
      log.detour_kf.push_back(std::numeric_limits<double>::quiet_NaN());
      log.detour_obs_idx.push_back(-1);
    }
    log.solve_time_s.push_back(last_solve_time_s_);
    log.wp_index.push_back(cur_idx);

    if (is_last) {
      const double dist_last = std::hypot(x - Xf, y - Yf);
      bool heading_last_ok = true;
      if (opts.use_heading_gate && wp.psig.has_value()) {
        heading_last_ok = std::abs(wrap_to_pi_np(psi - psig)) < opts.tol_psi;
      }
      if (dist_last <= tol_here && heading_last_ok) {
        break;
      }
    }
  }

  const Waypoint& wp_last = waypoints[std::min(cur_idx, static_cast<int>(waypoints.size()) - 1)];
  const double psig_last = pick_heading(x, y, wp_last);
  const double Kf_last = pick_Kf(wp_last);

  log.goal = State4{wp_last.X, wp_last.Y, psig_last, Kf_last};
  log.active_wp = cur_idx;
  if (!log.solve_time_s.empty()) {
    const double sum = std::accumulate(log.solve_time_s.begin(), log.solve_time_s.end(), 0.0);
    log.mean_solve_time_s = sum / static_cast<double>(log.solve_time_s.size());
  }

  return log;
}

void MPCNumericClothoidCost::write_log_csv(const RecedingLog& log, const std::string& path) const {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + path);
  }

  ofs << "step,x,y,psi,K,Kcmd,ds,wp_index,detour_wp_x,detour_wp_y,detour_kf,detour_obs_idx\n";
  const int n_state = static_cast<int>(log.traj.size());
  for (int i = 0; i < n_state; ++i) {
    const double x = log.traj[i].first;
    const double y = log.traj[i].second;
    const double psi = (i < static_cast<int>(log.psi.size())) ? log.psi[i] : 0.0;
    const double K = (i < static_cast<int>(log.K.size())) ? log.K[i] : 0.0;
    const double Kcmd = (i > 0 && (i - 1) < static_cast<int>(log.Kcmd.size())) ? log.Kcmd[i - 1] : 0.0;
    const double ds = (i > 0 && (i - 1) < static_cast<int>(log.ds.size())) ? log.ds[i - 1] : 0.0;
    const int wp_idx = (i < static_cast<int>(log.wp_index.size())) ? log.wp_index[i] : -1;
    const double detour_x =
        (i < static_cast<int>(log.detour_wp_x.size())) ? log.detour_wp_x[i] : std::numeric_limits<double>::quiet_NaN();
    const double detour_y =
        (i < static_cast<int>(log.detour_wp_y.size())) ? log.detour_wp_y[i] : std::numeric_limits<double>::quiet_NaN();
    const double detour_kf =
        (i < static_cast<int>(log.detour_kf.size())) ? log.detour_kf[i] : std::numeric_limits<double>::quiet_NaN();
    const int detour_obs = (i < static_cast<int>(log.detour_obs_idx.size())) ? log.detour_obs_idx[i] : -1;
    ofs << i << ',' << x << ',' << y << ',' << psi << ',' << K << ',' << Kcmd << ',' << ds << ',' << wp_idx << ','
        << detour_x << ',' << detour_y << ',' << detour_kf << ',' << detour_obs << '\n';
  }
}
