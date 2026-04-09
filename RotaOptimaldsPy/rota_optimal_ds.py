from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import casadi as ca
except ModuleNotFoundError as exc:
    ca = None
    _CASADI_IMPORT_ERROR = exc
else:
    _CASADI_IMPORT_ERROR = None

import numpy as np


@dataclass
class Waypoint:
    X: float = 0.0
    Y: float = 0.0
    psig: Optional[float] = None
    Kf: Optional[float] = None
    tol: Optional[float] = None
    w_wp: Optional[float] = None
    hit_scale: Optional[float] = None
    use_Kf: bool = True


@dataclass
class CircleObstacle:
    cx: float = 0.0
    cy: float = 0.0
    radius: float = 0.0
    enabled: bool = True


@dataclass
class State4:
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0
    K: float = 0.0


@dataclass
class MPCConfig:
    N: int = 25
    ds_min: float = 0.01
    ds_max: float = 1.0
    K_MAX: float = 0.30
    S_MAX: float = 14.0
    nseg: int = 4
    w_pos: float = 50.0
    w_psi: float = 250.0
    w_K: float = 0.5
    w_Kcmd: float = 0.5
    w_dKcmd: float = 2.0
    w_ds_smooth: float = 1.0
    ds_jump_max: Optional[float] = None
    w_Kf: float = 10.0
    enable_terminal_K_hard: bool = False
    ipopt_max_iter: int = 2000
    ipopt_tol: float = 1e-6
    block_lengths_Kcmd: list[int] = field(default_factory=list)
    block_lengths_ds: list[int] = field(default_factory=list)
    w_prog: float = 0.0
    alpha_prog: float = 0.0
    hit_ratio: float = 0.7


@dataclass
class MPCSolution:
    X: list[float]
    Kcmd: list[float]
    ds: list[float]
    start: State4
    goal: State4
    xwp: float = 0.0
    ywp: float = 0.0
    w_wp: float = 0.0
    term_scale: float = 1.0


@dataclass
class StepOutput:
    state: State4
    ds0: float
    Kcmd0: float
    sol: MPCSolution


@dataclass
class RecedingOptions:
    tol_default: float = 2.5
    max_iters: int = 300
    use_heading_gate: bool = False
    tol_psi: float = 5.0 * math.pi / 180.0
    w_wp_intermediate: float = 0.0
    term_scale_intermediate: float = 0.1
    term_scale_final: float = 1.0
    hit_scale_intermediate: float = 1.0
    w_wp_final: float = 2.0
    use_wp_kf: bool = True
    kf_fallback: float = 0.0
    enable_obstacle_avoidance: bool = False
    obstacle_clearance: float = 0.5
    obstacle_trigger_margin: float = 0.5
    obstacle_waypoint_tol: float = 1.5
    obstacles: list[CircleObstacle] = field(default_factory=list)


@dataclass
class RecedingLog:
    traj: list[tuple[float, float]] = field(default_factory=list)
    psi: list[float] = field(default_factory=list)
    K: list[float] = field(default_factory=list)
    Kcmd: list[float] = field(default_factory=list)
    ds: list[float] = field(default_factory=list)
    detour_wp_x: list[float] = field(default_factory=list)
    detour_wp_y: list[float] = field(default_factory=list)
    detour_kf: list[float] = field(default_factory=list)
    detour_obs_idx: list[int] = field(default_factory=list)
    solve_time_s: list[float] = field(default_factory=list)
    mean_solve_time_s: float = 0.0
    start: State4 = field(default_factory=State4)
    goal: State4 = field(default_factory=State4)
    waypoints: list[Waypoint] = field(default_factory=list)
    wp_index: list[int] = field(default_factory=list)
    active_wp: int = 0


@dataclass
class WarmStartData:
    X: list[float] = field(default_factory=list)
    Kcmd: list[float] = field(default_factory=list)
    ds: list[float] = field(default_factory=list)
    valid: bool = False


def clip(value, lo, hi):
    return min(hi, max(lo, value))


def wp_tol(wp, tol_default):
    return wp.tol if wp.tol is not None else tol_default


class MPCNumericClothoidCost:
    def __init__(self, cfg):
        if ca is None:
            raise RuntimeError(
                "casadi is required for RotaOptimaldsPy. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from _CASADI_IMPORT_ERROR

        self.cfg = cfg
        if self.cfg.N < 2:
            raise RuntimeError("N must be >= 2")

        self.k_hit = int(round(self.cfg.hit_ratio * float(self.cfg.N)))
        self.k_hit = max(1, min(self.cfg.N - 1, self.k_hit))

        self.bl_kcmd: list[int] = []
        self.bl_ds: list[int] = []
        self.NBK = 0
        self.NBd = 0

        self.last_sol: Optional[MPCSolution] = None
        self.last_warm = WarmStartData(valid=False)
        self.last_ds_applied = self.cfg.ds_max
        self.last_solve_time_s = 0.0

        self.compute_block_maps()
        self.build_solver()

    @staticmethod
    def wrap_to_pi(a):
        if isinstance(a, (float, int)):
            return math.atan2(math.sin(a), math.cos(a))
        return ca.atan2(ca.sin(a), ca.cos(a))

    @staticmethod
    def sinc(x):
        return ca.if_else((x * x) < 1e-16, 1 - (x * x) / 6.0, ca.sin(x) / x)

    @staticmethod
    def wrap_to_pi_np(a):
        return math.atan2(math.sin(a), math.cos(a))

    @staticmethod
    def step_constK_sinc_np(x, y, psi, K, ds):
        dpsi = K * ds
        half = dpsi / 2.0
        fac = 1.0 - (half * half) / 6.0 if abs(half) < 1e-12 else math.sin(half) / half
        x1 = x + ds * fac * math.cos(psi + dpsi / 2.0)
        y1 = y + ds * fac * math.sin(psi + dpsi / 2.0)
        psi1 = psi + dpsi
        return x1, y1, psi1

    @classmethod
    def clothoid_increment_numeric_np(
        cls,
        x0,
        y0,
        psi0,
        K0,
        K1,
        ds,
        nseg,
    ):
        ds_seg = ds / float(nseg)
        x, y, psi = x0, y0, psi0
        for i in range(nseg):
            K_mid = K0 + (K1 - K0) * ((float(i) + 0.5) / float(nseg))
            x, y, psi = cls.step_constK_sinc_np(x, y, psi, K_mid, ds_seg)
        return x, y, psi

    @staticmethod
    def K_next_fixed_ramp_np(
        Kcur,
        Kcmd,
        ds,
        K_MAX,
        S_MAX,
        eps=1e-9,
    ):
        a0 = K_MAX / S_MAX
        delta = Kcmd - Kcur
        max_step = a0 * ds
        dK = max_step * math.tanh(delta / (max_step + eps))
        return Kcur + dK

    @classmethod
    def clothoid_increment_numeric(cls, x0, y0, psi0, K0, K1, ds, nseg):
        x = x0
        y = y0
        psi = psi0
        ds_seg = ds / float(nseg)
        for i in range(nseg):
            K_mid = K0 + (K1 - K0) * ((float(i) + 0.5) / float(nseg))
            dpsi = K_mid * ds_seg
            fac = cls.sinc(dpsi / (2.0 * math.pi))
            x1 = x + ds_seg * fac * ca.cos(psi + dpsi / 2.0)
            y1 = y + ds_seg * fac * ca.sin(psi + dpsi / 2.0)
            psi1 = psi + dpsi
            x, y, psi = x1, y1, psi1
        return x, y, psi

    @staticmethod
    def K_next_fixed_ramp(Kcur, Kcmd, ds, K_MAX, S_MAX, eps=1e-9):
        a0 = K_MAX / S_MAX
        delta = Kcmd - Kcur
        max_step = a0 * ds
        dK = max_step * ca.tanh(delta / (max_step + eps))
        return Kcur + dK

    @staticmethod
    def x_index(row, col):
        return col * 4 + row

    def compute_block_maps(self):
        def make_map(lengths):
            if not lengths:
                return [], 0
            total = sum(lengths)
            if total != self.cfg.N:
                raise RuntimeError(f"block_lengths sum must equal N (sum={total}, N={self.cfg.N})")
            block_map = [0] * self.cfg.N
            s = 0
            for idx, length in enumerate(lengths):
                for j in range(length):
                    block_map[s + j] = idx
                s += length
            return block_map, len(lengths)

        self.bl_kcmd, self.NBK = make_map(self.cfg.block_lengths_Kcmd)
        self.bl_ds, self.NBd = make_map(self.cfg.block_lengths_ds)

    def block_init_from_full(self, values, blk_map, n_blocks):
        out = [0.0] * n_blocks
        for i in range(n_blocks):
            vals = [values[k] for k, blk in enumerate(blk_map) if blk == i]
            out[i] = sum(vals) / float(len(vals)) if vals else values[0]
        return out

    def build_solver(self):
        N = self.cfg.N
        self.opti = ca.Opti()

        self.X = self.opti.variable(4, N + 1)
        self.Kcmd = self.opti.variable(1, N)
        self.ds = self.opti.variable(1, N)

        self.x0_p = self.opti.parameter()
        self.y0_p = self.opti.parameter()
        self.psi0_p = self.opti.parameter()
        self.K0_p = self.opti.parameter()
        self.xg_p = self.opti.parameter()
        self.yg_p = self.opti.parameter()
        self.psig_p = self.opti.parameter()
        self.Kf_p = self.opti.parameter()
        self.xhit_p = self.opti.parameter()
        self.yhit_p = self.opti.parameter()
        self.psihit_p = self.opti.parameter()
        self.Khit_p = self.opti.parameter()
        self.hit_scale_p = self.opti.parameter()
        self.ds_prev_p = self.opti.parameter()
        self.term_scale_p = self.opti.parameter()
        self.w_wp_p = self.opti.parameter()
        self.xwp_p = self.opti.parameter()
        self.ywp_p = self.opti.parameter()

        self.opti.subject_to(self.X[0, 0] == self.x0_p)
        self.opti.subject_to(self.X[1, 0] == self.y0_p)
        self.opti.subject_to(self.X[2, 0] == self.psi0_p)
        self.opti.subject_to(self.X[3, 0] == self.K0_p)

        self.KcmdB = None
        if self.bl_kcmd:
            self.KcmdB = self.opti.variable(1, self.NBK)
            self.opti.subject_to(self.opti.bounded(-self.cfg.K_MAX, self.KcmdB, self.cfg.K_MAX))
            for k in range(N):
                self.opti.subject_to(self.Kcmd[0, k] == self.KcmdB[0, self.bl_kcmd[k]])
        else:
            self.opti.subject_to(self.opti.bounded(-self.cfg.K_MAX, self.Kcmd, self.cfg.K_MAX))

        self.opti.subject_to(self.opti.bounded(-self.cfg.K_MAX, self.X[3, :], self.cfg.K_MAX))

        self.dsB = None
        if self.bl_ds:
            self.dsB = self.opti.variable(1, self.NBd)
            self.opti.subject_to(self.opti.bounded(self.cfg.ds_min, self.dsB, self.cfg.ds_max))
            for k in range(N):
                self.opti.subject_to(self.ds[0, k] == self.dsB[0, self.bl_ds[k]])
        else:
            self.opti.subject_to(self.opti.bounded(self.cfg.ds_min, self.ds, self.cfg.ds_max))

        if self.cfg.ds_jump_max is not None and self.cfg.ds_jump_max > 0.0:
            jmax = abs(self.cfg.ds_jump_max)
            for k in range(1, N):
                self.opti.subject_to(self.opti.bounded(-jmax, self.ds[0, k] - self.ds[0, k - 1], jmax))
            self.opti.subject_to(self.opti.bounded(-jmax, self.ds[0, 0] - self.ds_prev_p, jmax))

        for k in range(N):
            ds_k = self.ds[0, k]
            K1 = self.K_next_fixed_ramp(self.X[3, k], self.Kcmd[0, k], ds_k, self.cfg.K_MAX, self.cfg.S_MAX)
            x1, y1, psi1 = self.clothoid_increment_numeric(
                self.X[0, k],
                self.X[1, k],
                self.X[2, k],
                self.X[3, k],
                K1,
                ds_k,
                self.cfg.nseg,
            )
            self.opti.subject_to(self.X[3, k + 1] == K1)
            self.opti.subject_to(self.X[0, k + 1] == x1)
            self.opti.subject_to(self.X[1, k + 1] == y1)
            self.opti.subject_to(self.X[2, k + 1] == psi1)

        obj = 0
        for k in range(N):
            obj += self.cfg.w_K * (self.X[3, k] ** 2)
            obj += self.cfg.w_Kcmd * (self.Kcmd[0, k] ** 2)
            if k > 0:
                obj += self.cfg.w_dKcmd * ((self.Kcmd[0, k] - self.Kcmd[0, k - 1]) ** 2)
                obj += self.cfg.w_ds_smooth * ((self.ds[0, k] - self.ds[0, k - 1]) ** 2)

        Dref2 = (self.xg_p - self.x0_p) ** 2 + (self.yg_p - self.y0_p) ** 2
        pos_scale = Dref2 + 1.0

        pos_e = (self.X[0, N] - self.xg_p) ** 2 + (self.X[1, N] - self.yg_p) ** 2
        psi_e = self.wrap_to_pi(self.X[2, N] - self.psig_p)
        K_e = self.X[3, N] - self.Kf_p
        obj += self.term_scale_p * (self.cfg.w_pos * pos_e / pos_scale)
        obj += self.term_scale_p * (self.cfg.w_psi * (psi_e ** 2))
        obj += self.term_scale_p * (self.cfg.w_Kf * (K_e ** 2))

        d2_sum = 0
        for k in range(1, N + 1):
            d2_sum += (self.X[0, k] - self.xwp_p) ** 2 + (self.X[1, k] - self.ywp_p) ** 2
        obj += self.w_wp_p * (d2_sum / pos_scale)

        kh = self.k_hit
        pos_hit = (self.X[0, kh] - self.xhit_p) ** 2 + (self.X[1, kh] - self.yhit_p) ** 2
        psi_hit = self.wrap_to_pi(self.X[2, kh] - self.psihit_p)
        obj += self.hit_scale_p * (self.cfg.w_pos * pos_hit / pos_scale)
        obj += self.hit_scale_p * (self.cfg.w_psi * (psi_hit ** 2))
        obj += self.hit_scale_p * (self.cfg.w_Kf * ((self.X[3, kh] - self.Khit_p) ** 2))

        if self.cfg.enable_terminal_K_hard:
            self.opti.subject_to(self.X[3, N] == self.Kf_p)

        self.opti.minimize(obj)
        self.configure_ipopt()

    def configure_ipopt(self):
        p_opts = {"print_time": True}
        s_opts = {
            "max_iter": self.cfg.ipopt_max_iter,
            "print_level": 1,
            "tol": self.cfg.ipopt_tol,
            "print_timing_statistics": "yes",
        }
        self.opti.solver("ipopt", p_opts, s_opts)

    def configure_sqpmethod(self):
        sqp_opts = {
            "print_time": True,
            "max_iter": self.cfg.ipopt_max_iter,
            "print_header": False,
            "print_iteration": False,
            "tol_pr": self.cfg.ipopt_tol,
            "tol_du": self.cfg.ipopt_tol,
            "error_on_fail": False,
            "qpsol": "qrqp",
            "qpsol_options": {
                "print_iter": False,
                "print_header": False,
                "error_on_fail": False,
            },
        }
        self.opti.solver("sqpmethod", sqp_opts)

    def set_params(
        self,
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
        xwp,
        ywp,
        xhit,
        yhit,
        psihit,
        Khit,
        hit_scale,
        ds_prev,
    ):
        self.opti.set_value(self.x0_p, x0)
        self.opti.set_value(self.y0_p, y0)
        self.opti.set_value(self.psi0_p, psi0)
        self.opti.set_value(self.K0_p, K0)
        self.opti.set_value(self.xg_p, xg)
        self.opti.set_value(self.yg_p, yg)
        self.opti.set_value(self.psig_p, psig)
        self.opti.set_value(self.Kf_p, Kf)
        self.opti.set_value(self.xhit_p, xhit)
        self.opti.set_value(self.yhit_p, yhit)
        self.opti.set_value(self.psihit_p, psihit)
        self.opti.set_value(self.Khit_p, Khit)
        self.opti.set_value(self.hit_scale_p, hit_scale)
        self.opti.set_value(self.ds_prev_p, ds_prev)
        self.opti.set_value(self.term_scale_p, term_scale)
        self.opti.set_value(self.w_wp_p, w_wp)
        self.opti.set_value(self.xwp_p, xwp)
        self.opti.set_value(self.ywp_p, ywp)

    def warm_start(
        self,
        x0,
        y0,
        psi0,
        K0,
        xg,
        yg,
        psig,
        ds_seed,
    ):
        dist = math.hypot(xg - x0, yg - y0)
        dpsi_target = self.wrap_to_pi_np(psig - psi0)
        heading_only_turn = dist < 1e-3 and abs(dpsi_target) > 0.25
        psi_goal = psig if heading_only_turn else math.atan2(yg - y0, xg - x0)
        dpsi_goal = dpsi_target if heading_only_turn else self.wrap_to_pi_np(psi_goal - psi0)
        heading_span = max(abs(dpsi_target), 1e-3)
        kcmd_den = 0.5 * float(self.cfg.N) * self.cfg.ds_max if heading_only_turn else max(dist, 1e-3)
        Kcmd_guess = clip(dpsi_goal / max(kcmd_den, 1e-3), -self.cfg.K_MAX, self.cfg.K_MAX)
        Kcmd_ws = [Kcmd_guess] * self.cfg.N

        if ds_seed is not None:
            ds_guess = clip(ds_seed, self.cfg.ds_min, self.cfg.ds_max)
        else:
            if heading_only_turn:
                ds_turn = heading_span / (0.75 * float(self.cfg.N) * max(self.cfg.K_MAX, 1e-6))
                ds_guess = clip(ds_turn, self.cfg.ds_min, self.cfg.ds_max)
            else:
                ds_guess = clip(dist / max(self.cfg.N, 1), self.cfg.ds_min, self.cfg.ds_max)
            if dist > 2.0:
                ds_guess = max(ds_guess, 0.6 * self.cfg.ds_max)
        ds_ws = [ds_guess] * self.cfg.N

        x_ws = [0.0] * (self.cfg.N + 1)
        y_ws = [0.0] * (self.cfg.N + 1)
        psi_ws = [0.0] * (self.cfg.N + 1)
        K_ws = [0.0] * (self.cfg.N + 1)
        x_ws[0], y_ws[0], psi_ws[0], K_ws[0] = x0, y0, psi0, K0

        for k in range(self.cfg.N):
            K_ws[k + 1] = self.K_next_fixed_ramp_np(
                K_ws[k],
                Kcmd_ws[k],
                ds_ws[k],
                self.cfg.K_MAX,
                self.cfg.S_MAX,
            )
            x_ws[k + 1], y_ws[k + 1], psi_ws[k + 1] = self.clothoid_increment_numeric_np(
                x_ws[k],
                y_ws[k],
                psi_ws[k],
                K_ws[k],
                K_ws[k + 1],
                ds_ws[k],
                self.cfg.nseg,
            )

        for k in range(self.cfg.N):
            self.opti.set_initial(self.Kcmd[0, k], Kcmd_ws[k])
            self.opti.set_initial(self.ds[0, k], ds_ws[k])
        for k in range(self.cfg.N + 1):
            self.opti.set_initial(self.X[0, k], x_ws[k])
            self.opti.set_initial(self.X[1, k], y_ws[k])
            self.opti.set_initial(self.X[2, k], psi_ws[k])
            self.opti.set_initial(self.X[3, k], K_ws[k])

        if self.KcmdB is not None:
            for idx, value in enumerate(self.block_init_from_full(Kcmd_ws, self.bl_kcmd, self.NBK)):
                self.opti.set_initial(self.KcmdB[0, idx], value)
        if self.dsB is not None:
            for idx, value in enumerate(self.block_init_from_full(ds_ws, self.bl_ds, self.NBd)):
                self.opti.set_initial(self.dsB[0, idx], value)

    def apply_warm_start(
        self,
        x0,
        y0,
        psi0,
        K0,
        xg,
        yg,
        psig,
        use_last_warm,
        ds_seed,
    ):
        if use_last_warm and self.last_warm.valid:
            for k in range(self.cfg.N):
                self.opti.set_initial(self.Kcmd[0, k], self.last_warm.Kcmd[k])
                self.opti.set_initial(self.ds[0, k], self.last_warm.ds[k])
            for k in range(self.cfg.N + 1):
                self.opti.set_initial(self.X[0, k], self.last_warm.X[self.x_index(0, k)])
                self.opti.set_initial(self.X[1, k], self.last_warm.X[self.x_index(1, k)])
                self.opti.set_initial(self.X[2, k], self.last_warm.X[self.x_index(2, k)])
                self.opti.set_initial(self.X[3, k], self.last_warm.X[self.x_index(3, k)])
            if self.KcmdB is not None:
                for idx, value in enumerate(self.block_init_from_full(self.last_warm.Kcmd, self.bl_kcmd, self.NBK)):
                    self.opti.set_initial(self.KcmdB[0, idx], value)
            if self.dsB is not None:
                for idx, value in enumerate(self.block_init_from_full(self.last_warm.ds, self.bl_ds, self.NBd)):
                    self.opti.set_initial(self.dsB[0, idx], value)
            return

        self.warm_start(x0, y0, psi0, K0, xg, yg, psig, ds_seed)

    def shift_solution(self, sol):
        ws = WarmStartData(
            X=[0.0] * len(sol.X),
            Kcmd=[0.0] * len(sol.Kcmd),
            ds=[0.0] * len(sol.ds),
            valid=True,
        )

        for k in range(self.cfg.N):
            ws.X[self.x_index(0, k)] = sol.X[self.x_index(0, k + 1)]
            ws.X[self.x_index(1, k)] = sol.X[self.x_index(1, k + 1)]
            ws.X[self.x_index(2, k)] = sol.X[self.x_index(2, k + 1)]
            ws.X[self.x_index(3, k)] = sol.X[self.x_index(3, k + 1)]
        ws.X[self.x_index(0, self.cfg.N)] = sol.X[self.x_index(0, self.cfg.N)]
        ws.X[self.x_index(1, self.cfg.N)] = sol.X[self.x_index(1, self.cfg.N)]
        ws.X[self.x_index(2, self.cfg.N)] = sol.X[self.x_index(2, self.cfg.N)]
        ws.X[self.x_index(3, self.cfg.N)] = sol.X[self.x_index(3, self.cfg.N)]

        for k in range(self.cfg.N - 1):
            ws.Kcmd[k] = sol.Kcmd[k + 1]
            ws.ds[k] = sol.ds[k + 1]
        ws.Kcmd[self.cfg.N - 1] = sol.Kcmd[self.cfg.N - 1]
        ws.ds[self.cfg.N - 1] = sol.ds[self.cfg.N - 1]
        self.last_warm = ws

    def solve(
        self,
        x0,
        y0,
        psi0,
        K0,
        xg,
        yg,
        psig,
        Kf,
        term_scale=1.0,
        w_wp=0.0,
        xwp=None,
        ywp=None,
        hit_scale=0.0,
        xhit=None,
        yhit=None,
        psihit=None,
        Khit=None,
        ds_prev=None,
        ds_seed=None,
        use_last_warm=True,
    ):
        xwp_v = xg if xwp is None else xwp
        ywp_v = yg if ywp is None else ywp
        xhit_v = xg if xhit is None else xhit
        yhit_v = yg if yhit is None else yhit
        psihit_v = psig if psihit is None else psihit
        Khit_v = Kf if Khit is None else Khit
        ds_prev_v = self.last_ds_applied if ds_prev is None else ds_prev

        self.set_params(
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
        )

        self.apply_warm_start(x0, y0, psi0, K0, xg, yg, psig, use_last_warm, ds_seed)

        t0 = time.perf_counter()
        sol = None
        try:
            sol = self.opti.solve()
        except Exception as exc:
            msg = str(exc)
            ipopt_missing = "Plugin 'ipopt' is not found" in msg or "libcasadi_nlpsol_ipopt" in msg
            restoration_failed = "Restoration_Failed" in msg

            if (not ipopt_missing) and use_last_warm:
                print("[WARN] IPOPT solve failed, retrying with fresh warm start.")
                self.apply_warm_start(x0, y0, psi0, K0, xg, yg, psig, False, ds_seed)
                try:
                    sol = self.opti.solve()
                except Exception as retry_exc:
                    retry_msg = str(retry_exc)
                    if (not restoration_failed) and ("Restoration_Failed" not in retry_msg) and (not ipopt_missing):
                        raise
                    print("[WARN] Fresh IPOPT retry failed, falling back to sqpmethod.")

            if sol is None:
                if ipopt_missing:
                    print("[WARN] IPOPT not available, falling back to sqpmethod.")
                elif restoration_failed or use_last_warm:
                    print("[WARN] Switching to sqpmethod fallback for this step.")
                else:
                    raise

                self.configure_sqpmethod()
                self.apply_warm_start(x0, y0, psi0, K0, xg, yg, psig, False, ds_seed)
                sol = self.opti.solve()
                self.configure_ipopt()

        dt = time.perf_counter() - t0
        self.last_solve_time_s = dt

        out = MPCSolution(
            X=np.array(sol.value(self.X), dtype=float).reshape(-1, order="F").tolist(),
            Kcmd=np.array(sol.value(self.Kcmd), dtype=float).reshape(-1).tolist(),
            ds=np.array(sol.value(self.ds), dtype=float).reshape(-1).tolist(),
            start=State4(x0, y0, psi0, K0),
            goal=State4(xg, yg, psig, Kf),
            xwp=xwp_v,
            ywp=ywp_v,
            w_wp=w_wp,
            term_scale=term_scale,
        )

        print(f"CasADi solve time: {dt:.3f} s")
        self.last_sol = out
        return out

    def mpc_step(
        self,
        state,
        goal,
        term_scale=1.0,
        w_wp=0.0,
        xwp=None,
        ywp=None,
        hit_scale=0.0,
        xhit=None,
        yhit=None,
        psihit=None,
        Khit=None,
        ds_seed=None,
    ):
        sol = self.solve(
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
            self.last_ds_applied,
            ds_seed,
            True,
        )

        ds0 = sol.ds[0]
        K1 = sol.X[self.x_index(3, 1)]
        x1, y1, psi1 = self.clothoid_increment_numeric_np(
            state.x,
            state.y,
            state.psi,
            state.K,
            K1,
            ds0,
            self.cfg.nseg,
        )
        self.last_ds_applied = ds0
        self.shift_solution(sol)
        return StepOutput(
            state=State4(x1, y1, psi1, K1),
            ds0=ds0,
            Kcmd0=sol.Kcmd[0],
            sol=sol,
        )

    def run_receding_horizon_multi(
        self,
        waypoints,
        initial_state,
        opts=None,
    ):
        if not waypoints:
            raise RuntimeError("Waypoint list cannot be empty.")
        if opts is None:
            opts = RecedingOptions()

        from obstacle_avoidance import select_obstacle_detour_waypoint

        def pick_heading(x, y, wp):
            if wp.psig is None:
                return math.atan2(wp.Y - y, wp.X - x)
            return wp.psig

        def pick_Kf(wp):
            if (not opts.use_wp_kf) or (not wp.use_Kf):
                return opts.kf_fallback
            if wp.Kf is None:
                return opts.kf_fallback
            return wp.Kf

        self.last_warm.valid = False
        self.last_ds_applied = self.cfg.ds_max

        state = State4(initial_state.x, initial_state.y, initial_state.psi, initial_state.K)
        x, y, psi, K = state.x, state.y, state.psi, state.K

        nan = float("nan")
        log = RecedingLog()
        log.traj.append((x, y))
        log.psi.append(psi)
        log.K.append(K)
        log.detour_wp_x.append(nan)
        log.detour_wp_y.append(nan)
        log.detour_kf.append(nan)
        log.detour_obs_idx.append(-1)
        log.start = initial_state
        log.waypoints = list(waypoints)
        log.wp_index.append(0)

        ds_seed_next: Optional[float] = None
        cur_idx = 0
        obstacle_done_for_wp = [False] * len(opts.obstacles)
        detour_wp: Optional[Waypoint] = None
        detour_obstacle_idx = -1

        for it in range(opts.max_iters):
            wp_main = waypoints[cur_idx]

            if detour_wp is None and opts.enable_obstacle_avoidance and opts.obstacles:
                detour = select_obstacle_detour_waypoint(
                    x,
                    y,
                    psi,
                    wp_main,
                    opts.obstacles,
                    obstacle_done_for_wp,
                    opts.obstacle_clearance,
                    opts.obstacle_trigger_margin,
                    opts.obstacle_waypoint_tol,
                )
                if detour is not None:
                    detour_wp = detour.waypoint
                    detour_obstacle_idx = detour.obstacle_index
                    self.last_warm.valid = False
                    ds_seed_next = log.ds[-1] if log.ds else None
                    detour_kf = detour_wp.Kf if detour_wp.Kf is not None else opts.kf_fallback
                    print(
                        f"[INFO] Obstacle detour activated (obs={detour_obstacle_idx}) "
                        f"via [{detour_wp.X}, {detour_wp.Y}], Kf={detour_kf}"
                    )

            using_detour = detour_wp is not None
            wp = detour_wp if using_detour else wp_main
            Xf = wp.X
            Yf = wp.Y
            psig = pick_heading(x, y, wp)
            Kf = pick_Kf(wp)
            tol_here = wp_tol(wp, opts.tol_default)
            is_last = (not using_detour) and (cur_idx >= len(waypoints) - 1)

            dist_now = math.hypot(x - Xf, y - Yf)
            heading_ok = True
            if opts.use_heading_gate and (not using_detour) and wp.psig is not None:
                heading_ok = abs(self.wrap_to_pi_np(psi - psig)) < opts.tol_psi

            if using_detour and dist_now <= tol_here:
                if 0 <= detour_obstacle_idx < len(obstacle_done_for_wp):
                    obstacle_done_for_wp[detour_obstacle_idx] = True
                detour_wp = None
                detour_obstacle_idx = -1
                self.last_warm.valid = False
                ds_seed_next = log.ds[-1] if log.ds else None
                continue

            if (not using_detour) and (not is_last) and dist_now <= tol_here and heading_ok:
                self.last_warm.valid = False
                ds_seed_next = log.ds[-1] if log.ds else None
                cur_idx += 1
                obstacle_done_for_wp = [False] * len(opts.obstacles)
                continue

            xwp = Xf
            ywp = Yf
            xhit = Xf
            yhit = Yf
            psihit = psig
            Khit = Kf
            if is_last:
                goal_from_start = math.hypot(wp_main.X - initial_state.x, wp_main.Y - initial_state.y)
                heading_from_start = abs(self.wrap_to_pi_np(psig - initial_state.psi))
                same_point_heading_turn = (goal_from_start <= tol_here) and (heading_from_start > opts.tol_psi)

                term_scale = opts.term_scale_final
                w_wp = 0.0 if same_point_heading_turn else opts.w_wp_final
                hit_scale = 0.0

                if same_point_heading_turn:
                    dpsi_total = self.wrap_to_pi_np(psig - initial_state.psi)
                    psi_mid = self.wrap_to_pi_np(initial_state.psi + 0.5 * dpsi_total)
                    loop_radius = max(2.0, 1.0 / max(self.cfg.K_MAX, 1e-6))
                    xhit = Xf + loop_radius * math.cos(psi_mid)
                    yhit = Yf + loop_radius * math.sin(psi_mid)
                    psihit = psi_mid
                    xwp = xhit
                    ywp = yhit
                    w_wp = max(0.5, opts.w_wp_final)
                    hit_scale = 0.75 * opts.term_scale_final
            else:
                term_scale = opts.term_scale_intermediate
                w_wp = wp.w_wp if wp.w_wp is not None else opts.w_wp_intermediate
                hit_scale = wp.hit_scale if wp.hit_scale is not None else opts.hit_scale_intermediate

            try:
                step = self.mpc_step(
                    state,
                    State4(Xf, Yf, psig, Kf),
                    term_scale,
                    w_wp,
                    xwp,
                    ywp,
                    hit_scale,
                    xhit,
                    yhit,
                    psihit,
                    Khit,
                    ds_seed_next,
                )
            except Exception as exc:
                print(f"[WARN] MPC step failed at iter {it}: {exc}")
                break

            ds_seed_next = None
            state = step.state
            x, y, psi, K = state.x, state.y, state.psi, state.K

            log.traj.append((x, y))
            log.psi.append(psi)
            log.K.append(K)
            log.Kcmd.append(step.Kcmd0)
            log.ds.append(step.ds0)
            if using_detour:
                log.detour_wp_x.append(Xf)
                log.detour_wp_y.append(Yf)
                log.detour_kf.append(Kf)
                log.detour_obs_idx.append(detour_obstacle_idx)
            else:
                log.detour_wp_x.append(nan)
                log.detour_wp_y.append(nan)
                log.detour_kf.append(nan)
                log.detour_obs_idx.append(-1)
            log.solve_time_s.append(self.last_solve_time_s)
            log.wp_index.append(cur_idx)

            if is_last:
                dist_last = math.hypot(x - Xf, y - Yf)
                heading_last_ok = True
                if opts.use_heading_gate and wp.psig is not None:
                    heading_last_ok = abs(self.wrap_to_pi_np(psi - psig)) < opts.tol_psi
                if dist_last <= tol_here and heading_last_ok:
                    break

        wp_last = waypoints[min(cur_idx, len(waypoints) - 1)]
        psig_last = pick_heading(x, y, wp_last)
        Kf_last = pick_Kf(wp_last)
        log.goal = State4(wp_last.X, wp_last.Y, psig_last, Kf_last)
        log.active_wp = cur_idx
        if log.solve_time_s:
            log.mean_solve_time_s = sum(log.solve_time_s) / float(len(log.solve_time_s))
        return log

    def write_log_csv(self, log, path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "x",
                    "y",
                    "psi",
                    "K",
                    "Kcmd",
                    "ds",
                    "wp_index",
                    "detour_wp_x",
                    "detour_wp_y",
                    "detour_kf",
                    "detour_obs_idx",
                ]
            )
            n_state = len(log.traj)
            for i in range(n_state):
                x, y = log.traj[i]
                psi = log.psi[i] if i < len(log.psi) else 0.0
                K = log.K[i] if i < len(log.K) else 0.0
                Kcmd = log.Kcmd[i - 1] if i > 0 and (i - 1) < len(log.Kcmd) else 0.0
                ds = log.ds[i - 1] if i > 0 and (i - 1) < len(log.ds) else 0.0
                wp_idx = log.wp_index[i] if i < len(log.wp_index) else -1
                detour_x = log.detour_wp_x[i] if i < len(log.detour_wp_x) else float("nan")
                detour_y = log.detour_wp_y[i] if i < len(log.detour_wp_y) else float("nan")
                detour_kf = log.detour_kf[i] if i < len(log.detour_kf) else float("nan")
                detour_obs = log.detour_obs_idx[i] if i < len(log.detour_obs_idx) else -1
                writer.writerow([i, x, y, psi, K, Kcmd, ds, wp_idx, detour_x, detour_y, detour_kf, detour_obs])
