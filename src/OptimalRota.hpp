# pragma once 

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

struct CicrleObstacle {
    double cx = 0.0;
    double cy = 0.0;
    double radius = 0.0;
    bool enabled = true;

};

struct State4 {
    double  x= 0.0;
    double  y= 0.0;
    double  psi = 0.0;
    double  K = 0.0;
};

struct MPCConfig {
    int N = 25;
    double ds_min = 0.01;
    double ds_max = 0.3;
    double K_MAX  = 0.3;
    double S_MAX  = 14.0;
    int nseg = 4;

    double w_pos = 50.0;
    double w_psi =250;
    double w_K = 0.5;
    double w_Kcmd = 0.5;
    double w_dKcmd = 2.0;
    double w_ds_smooth = 1.0;
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
