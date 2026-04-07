#include "rota_optimal_ds.hpp"
#include "obstacle_avoidance.hpp"

#include <algorithm>
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
double clip(double v, double lo, double hi) {
  return std::min(hi, std::max(lo, v));
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

std::vector<double> MPCNumericClothoidCost::block_init_from_full(
    const std::vector<double>& values,
    const std::vector<int>& blk_map,
    int n_blocks) const {
  std::vector<double> out(n_blocks, 0.0);
  for (int i = 0; i < n_blocks; ++i) {
    double sum = 0.0;
    int cnt = 0;
    for (int k = 0; k < static_cast<int>(blk_map.size()); ++k) {
      if (blk_map[k] == i) {
        sum += values[k];
        ++cnt;
      }
    }
    out[i] = (cnt > 0) ? (sum / static_cast<double>(cnt)) : values.front();
  }
  return out;
}

void MPCNumericClothoidCost::build_solver() {
  const int N = cfg_.N;

  X_ = opti_.variable(4, N + 1);
  Kcmd_ = opti_.variable(1, N);
  ds_ = opti_.variable(1, N);

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

  opti_.subject_to(X_(0, 0) == x0_p_);
  opti_.subject_to(X_(1, 0) == y0_p_);
  opti_.subject_to(X_(2, 0) == psi0_p_);
  opti_.subject_to(X_(3, 0) == K0_p_);

  if (!bl_kcmd_.empty()) {
    KcmdB_ = opti_.variable(1, NBK_);
    opti_.subject_to(opti_.bounded(-cfg_.K_MAX, *KcmdB_, cfg_.K_MAX));
    for (int k = 0; k < N; ++k) {
      opti_.subject_to(Kcmd_(0, k) == (*KcmdB_)(0, bl_kcmd_[k]));
    }
  } else {
    opti_.subject_to(opti_.bounded(-cfg_.K_MAX, Kcmd_, cfg_.K_MAX));
  }

  opti_.subject_to(opti_.bounded(-cfg_.K_MAX, X_(3, casadi::Slice(0, N + 1)), cfg_.K_MAX));

  if (!bl_ds_.empty()) {
    dsB_ = opti_.variable(1, NBd_);
    opti_.subject_to(opti_.bounded(cfg_.ds_min, *dsB_, cfg_.ds_max));
    for (int k = 0; k < N; ++k) {
      opti_.subject_to(ds_(0, k) == (*dsB_)(0, bl_ds_[k]));
    }
  } else {
    opti_.subject_to(opti_.bounded(cfg_.ds_min, ds_, cfg_.ds_max));
  }

  if (cfg_.ds_jump_max.has_value() && *cfg_.ds_jump_max > 0.0) {
    const double jmax = std::abs(*cfg_.ds_jump_max);
    for (int k = 1; k < N; ++k) {
      opti_.subject_to(opti_.bounded(-jmax, ds_(0, k) - ds_(0, k - 1), jmax));
    }
    opti_.subject_to(opti_.bounded(-jmax, ds_(0, 0) - ds_prev_p_, jmax));
  }

  for (int k = 0; k < N; ++k) {
    const MX ds_k = ds_(0, k);
    const MX K1 = K_next_fixed_ramp(X_(3, k), Kcmd_(0, k), ds_k, cfg_.K_MAX, cfg_.S_MAX);
    MX x1, y1, psi1;
    std::tie(x1, y1, psi1) =
        clothoid_increment_numeric(X_(0, k), X_(1, k), X_(2, k), X_(3, k), K1, ds_k, cfg_.nseg);
    opti_.subject_to(X_(3, k + 1) == K1);
    opti_.subject_to(X_(0, k + 1) == x1);
    opti_.subject_to(X_(1, k + 1) == y1);
    opti_.subject_to(X_(2, k + 1) == psi1);
  }

  MX obj = 0;
  for (int k = 0; k < N; ++k) {
    obj += cfg_.w_K * (X_(3, k) * X_(3, k));
    obj += cfg_.w_Kcmd * (Kcmd_(0, k) * Kcmd_(0, k));
    if (k > 0) {
      obj += cfg_.w_dKcmd * ((Kcmd_(0, k) - Kcmd_(0, k - 1)) * (Kcmd_(0, k) - Kcmd_(0, k - 1)));
      obj += cfg_.w_ds_smooth * ((ds_(0, k) - ds_(0, k - 1)) * (ds_(0, k) - ds_(0, k - 1)));
    }
  }

  const double epsD = 1e-6;
  const MX Dref = sqrt((xg_p_ - x0_p_) * (xg_p_ - x0_p_) + (yg_p_ - y0_p_) * (yg_p_ - y0_p_));
  const MX Dref_safe = Dref + 1e-3;

  const MX pos_e = (X_(0, N) - xg_p_) * (X_(0, N) - xg_p_) + (X_(1, N) - yg_p_) * (X_(1, N) - yg_p_);
  const MX psi_e = wrap_to_pi(X_(2, N) - psig_p_);
  const MX K_e = X_(3, N) - Kf_p_;
  obj += term_scale_p_ * (cfg_.w_pos * pos_e / (Dref_safe * Dref_safe + epsD));
  obj += term_scale_p_ * (cfg_.w_psi * (psi_e * psi_e));
  obj += term_scale_p_ * (cfg_.w_Kf * (K_e * K_e));

  MX d2_sum = 0;
  for (int k = 1; k <= N; ++k) {
    d2_sum += (X_(0, k) - xwp_p_) * (X_(0, k) - xwp_p_) + (X_(1, k) - ywp_p_) * (X_(1, k) - ywp_p_);
  }
  obj += w_wp_p_ * (d2_sum / (Dref_safe * Dref_safe + epsD));

  const int kh = k_hit_;
  const MX pos_hit = (X_(0, kh) - xhit_p_) * (X_(0, kh) - xhit_p_) +
                     (X_(1, kh) - yhit_p_) * (X_(1, kh) - yhit_p_);
  const MX psi_hit = wrap_to_pi(X_(2, kh) - psihit_p_);
  obj += hit_scale_p_ * (cfg_.w_pos * pos_hit / (Dref_safe * Dref_safe + epsD));
  obj += hit_scale_p_ * (cfg_.w_psi * (psi_hit * psi_hit));
  obj += hit_scale_p_ * (cfg_.w_Kf * ((X_(3, kh) - Khit_p_) * (X_(3, kh) - Khit_p_)));

  if (cfg_.enable_terminal_K_hard) {
    opti_.subject_to(X_(3, N) == Kf_p_);
  }

  opti_.minimize(obj);

  casadi::Dict p_opts;
  p_opts["print_time"] = true;

  casadi::Dict s_opts;
  s_opts["max_iter"] = cfg_.ipopt_max_iter;
  s_opts["print_level"] = 1;
  s_opts["tol"] = cfg_.ipopt_tol;
  s_opts["print_timing_statistics"] = "yes";

  opti_.solver("ipopt", p_opts, s_opts);
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

void MPCNumericClothoidCost::warm_start(
    double x0,
    double y0,
    double psi0,
    double K0,
    double xg,
    double yg,
    std::optional<double> ds_seed) {
  const double dist = std::hypot(xg - x0, yg - y0);
  const double psi_goal = std::atan2(yg - y0, xg - x0);
  const double dpsi_goal = wrap_to_pi_np(psi_goal - psi0);
  const double Kcmd_guess = clip(dpsi_goal / std::max(dist, 1e-3), -cfg_.K_MAX, cfg_.K_MAX);

  std::vector<double> Kcmd_ws(cfg_.N, Kcmd_guess);

  double ds_guess = 0.0;
  if (ds_seed.has_value()) {
    ds_guess = clip(*ds_seed, cfg_.ds_min, cfg_.ds_max);
  } else {
    ds_guess = clip(dist / std::max(cfg_.N, 1), cfg_.ds_min, cfg_.ds_max);
    if (dist > 2.0) {
      ds_guess = std::max(ds_guess, 0.6 * cfg_.ds_max);
    }
  }
  std::vector<double> ds_ws(cfg_.N, ds_guess);

  std::vector<double> x_ws(cfg_.N + 1, 0.0);
  std::vector<double> y_ws(cfg_.N + 1, 0.0);
  std::vector<double> psi_ws(cfg_.N + 1, 0.0);
  std::vector<double> K_ws(cfg_.N + 1, 0.0);
  x_ws[0] = x0;
  y_ws[0] = y0;
  psi_ws[0] = psi0;
  K_ws[0] = K0;

  for (int k = 0; k < cfg_.N; ++k) {
    K_ws[k + 1] = K_next_fixed_ramp_np(K_ws[k], Kcmd_ws[k], ds_ws[k], cfg_.K_MAX, cfg_.S_MAX);
    std::tie(x_ws[k + 1], y_ws[k + 1], psi_ws[k + 1]) =
        clothoid_increment_numeric_np(x_ws[k], y_ws[k], psi_ws[k], K_ws[k], K_ws[k + 1], ds_ws[k], cfg_.nseg);
  }

  for (int k = 0; k < cfg_.N; ++k) {
    opti_.set_initial(Kcmd_(0, k), Kcmd_ws[k]);
    opti_.set_initial(ds_(0, k), ds_ws[k]);
  }
  for (int k = 0; k <= cfg_.N; ++k) {
    opti_.set_initial(X_(0, k), x_ws[k]);
    opti_.set_initial(X_(1, k), y_ws[k]);
    opti_.set_initial(X_(2, k), psi_ws[k]);
    opti_.set_initial(X_(3, k), K_ws[k]);
  }

  if (KcmdB_.has_value()) {
    const auto b = block_init_from_full(Kcmd_ws, bl_kcmd_, NBK_);
    for (int i = 0; i < NBK_; ++i) {
      opti_.set_initial((*KcmdB_)(0, i), b[i]);
    }
  }
  if (dsB_.has_value()) {
    const auto b = block_init_from_full(ds_ws, bl_ds_, NBd_);
    for (int i = 0; i < NBd_; ++i) {
      opti_.set_initial((*dsB_)(0, i), b[i]);
    }
  }
}

void MPCNumericClothoidCost::apply_warm_start(
    double x0,
    double y0,
    double psi0,
    double K0,
    double xg,
    double yg,
    bool use_last_warm,
    std::optional<double> ds_seed) {
  if (use_last_warm && last_warm_.valid) {
    for (int k = 0; k < cfg_.N; ++k) {
      opti_.set_initial(Kcmd_(0, k), last_warm_.Kcmd[k]);
      opti_.set_initial(ds_(0, k), last_warm_.ds[k]);
    }
    for (int k = 0; k <= cfg_.N; ++k) {
      opti_.set_initial(X_(0, k), last_warm_.X[x_index(0, k)]);
      opti_.set_initial(X_(1, k), last_warm_.X[x_index(1, k)]);
      opti_.set_initial(X_(2, k), last_warm_.X[x_index(2, k)]);
      opti_.set_initial(X_(3, k), last_warm_.X[x_index(3, k)]);
    }
    if (KcmdB_.has_value()) {
      const auto b = block_init_from_full(last_warm_.Kcmd, bl_kcmd_, NBK_);
      for (int i = 0; i < NBK_; ++i) {
        opti_.set_initial((*KcmdB_)(0, i), b[i]);
      }
    }
    if (dsB_.has_value()) {
      const auto b = block_init_from_full(last_warm_.ds, bl_ds_, NBd_);
      for (int i = 0; i < NBd_; ++i) {
        opti_.set_initial((*dsB_)(0, i), b[i]);
      }
    }
    return;
  }
  warm_start(x0, y0, psi0, K0, xg, yg, ds_seed);
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

  apply_warm_start(x0, y0, psi0, K0, xg, yg, use_last_warm, ds_seed);

  auto t0 = std::chrono::steady_clock::now();
  std::optional<casadi::OptiSol> sol;
  try {
    sol = opti_.solve();
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    if (msg.find("Plugin 'ipopt' is not found") != std::string::npos ||
        msg.find("libcasadi_nlpsol_ipopt") != std::string::npos) {
      std::cerr << "[WARN] IPOPT not available, falling back to sqpmethod.\n";
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
      sol = opti_.solve();
    } else {
      throw;
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  const std::chrono::duration<double> dt = t1 - t0;
  last_solve_time_s_ = dt.count();

  MPCSolution out;
  out.X = sol->value(X_).nonzeros();
  out.Kcmd = sol->value(Kcmd_).nonzeros();
  out.ds = sol->value(ds_).nonzeros();
  out.start = State4{x0, y0, psi0, K0};
  out.goal = State4{xg, yg, psig, Kf};
  out.xwp = xwp_v;
  out.ywp = ywp_v;
  out.w_wp = w_wp;
  out.term_scale = term_scale;

  std::cout << "CasADi solve time: " << std::fixed << std::setprecision(3) << dt.count() << " s\n";

  last_sol_ = out;
  return out;
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
    throw std::runtime_error("waypoints listesi bos olamaz.");
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
    if (opts.use_heading_gate && !using_detour) {
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
    if (is_last) {
      term_scale = opts.term_scale_final;
      w_wp = opts.w_wp_final;
      hit_scale = 0.0;
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
          Xf,
          Yf,
          hit_scale,
          Xf,
          Yf,
          psig,
          Kf,
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
      if (opts.use_heading_gate) {
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
