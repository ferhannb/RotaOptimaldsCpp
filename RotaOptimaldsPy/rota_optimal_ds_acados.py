from __future__ import annotations

import csv
import math
import os
import shutil
import time
from typing import Optional

import numpy as np

try:
    import casadi as ca
except ModuleNotFoundError as exc:
    ca = None
    _CASADI_IMPORT_ERROR = exc
else:
    _CASADI_IMPORT_ERROR = None

try:
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
except ModuleNotFoundError as exc:
    AcadosModel = None
    AcadosOcp = None
    AcadosOcpSolver = None
    _ACADOS_IMPORT_ERROR = exc
else:
    _ACADOS_IMPORT_ERROR = None

from rota_optimal_ds import (
    Waypoint,
    State4,
    MPCConfig,
    MPCSolution,
    StepOutput,
    RecedingOptions,
    RecedingLog,
    WarmStartData,
    clip,
    wp_tol,
)


class MPCAcadosClothoidCost:
    """
    acados backend for the same scenario/receding-horizon structure.

    State:
        [x, y, psi, K]

    Control:
        [Kcmd, ds]

    Internal augmented state:
        [x, y, psi, K, Kcmd_prev, ds_prev]

    Stage parameter vector p:
        [xg, yg, psig, Kf,
         xwp, ywp, w_wp,
         xhit, yhit, psihit, Khit, hit_scale,
         term_scale,
         x0_ref, y0_ref,
         smooth_scale,
         kcmd_hold_scale, ds_hold_scale]
    """

    NX = 4
    NX_AUG = 6
    NU = 2
    NP = 18

    IX_X = 0
    IX_Y = 1
    IX_PSI = 2
    IX_K = 3

    AIX_X = 0
    AIX_Y = 1
    AIX_PSI = 2
    AIX_K = 3
    AIX_KCMD_PREV = 4
    AIX_DS_PREV = 5

    IU_KCMD = 0
    IU_DS = 1

    def __init__(self, cfg: MPCConfig):
        if ca is None:
            raise RuntimeError("casadi is required for acados backend.") from _CASADI_IMPORT_ERROR
        if AcadosModel is None or AcadosOcp is None or AcadosOcpSolver is None:
            raise RuntimeError("acados_template is required for acados backend.") from _ACADOS_IMPORT_ERROR

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
        self.last_kcmd_applied = 0.0
        self.last_solve_time_s = 0.0

        self._solver: Optional[AcadosOcpSolver] = None
        self._ocp: Optional[AcadosOcp] = None

        self.compute_block_maps()
        self._build_solver()

    # -------------------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def wrap_to_pi_np(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))

    @staticmethod
    def wrap_to_pi(a):
        return ca.atan2(ca.sin(a), ca.cos(a))

    @staticmethod
    def sinc(x):
        return ca.if_else((x * x) < 1e-16, 1.0 - (x * x) / 6.0, ca.sin(x) / x)

    @classmethod
    def step_constK_sinc_np(cls, x, y, psi, K, ds):
        dpsi = K * ds
        half = 0.5 * dpsi
        fac = 1.0 - (half * half) / 6.0 if abs(half) < 1e-12 else math.sin(half) / half
        x1 = x + ds * fac * math.cos(psi + half)
        y1 = y + ds * fac * math.sin(psi + half)
        psi1 = psi + dpsi
        return x1, y1, psi1

    @classmethod
    def clothoid_increment_numeric_np(cls, x0, y0, psi0, K0, K1, ds, nseg):
        ds_seg = ds / float(nseg)
        x, y, psi = x0, y0, psi0
        for i in range(nseg):
            alpha = (float(i) + 0.5) / float(nseg)
            K_mid = K0 + (K1 - K0) * alpha
            x, y, psi = cls.step_constK_sinc_np(x, y, psi, K_mid, ds_seg)
        return x, y, psi

    @staticmethod
    def K_next_fixed_ramp_np(Kcur, Kcmd, ds, K_MAX, S_MAX, eps=1e-9):
        a0 = K_MAX / S_MAX
        delta = Kcmd - Kcur
        max_step = a0 * ds
        dK = max_step * math.tanh(delta / (max_step + eps))
        return Kcur + dK

    @staticmethod
    def K_next_fixed_ramp(Kcur, Kcmd, ds, K_MAX, S_MAX, eps=1e-9):
        a0 = K_MAX / S_MAX
        delta = Kcmd - Kcur
        max_step = a0 * ds
        dK = max_step * ca.tanh(delta / (max_step + eps))
        return Kcur + dK

    @classmethod
    def clothoid_increment_numeric(cls, x0, y0, psi0, K0, K1, ds, nseg):
        x = x0
        y = y0
        psi = psi0
        ds_seg = ds / float(nseg)

        for i in range(nseg):
            alpha = (float(i) + 0.5) / float(nseg)
            K_mid = K0 + (K1 - K0) * alpha
            dpsi = K_mid * ds_seg
            fac = cls.sinc(dpsi / (2.0 * math.pi))
            x = x + ds_seg * fac * ca.cos(psi + dpsi / 2.0)
            y = y + ds_seg * fac * ca.sin(psi + dpsi / 2.0)
            psi = psi + dpsi

        return x, y, psi

    @staticmethod
    def x_index(row: int, col: int) -> int:
        return col * 4 + row

    @staticmethod
    def _pack_aug_state(
        x: float,
        y: float,
        psi: float,
        K: float,
        kcmd_prev: float,
        ds_prev: float,
    ) -> np.ndarray:
        return np.array([x, y, psi, K, kcmd_prev, ds_prev], dtype=float)

    def compute_block_maps(self):
        def make_map(lengths):
            if not lengths:
                return [], 0
            total = sum(lengths)
            if total != self.cfg.N:
                raise RuntimeError(f"block_lengths sum must equal N (sum={total}, N={self.cfg.N})")
            block_map = [0] * self.cfg.N
            start = 0
            for idx, length in enumerate(lengths):
                for j in range(length):
                    block_map[start + j] = idx
                start += length
            return block_map, len(lengths)

        self.bl_kcmd, self.NBK = make_map(self.cfg.block_lengths_Kcmd)
        self.bl_ds, self.NBd = make_map(self.cfg.block_lengths_ds)

    @staticmethod
    def _hold_scale(block_map: list[int], k: int) -> float:
        if not block_map or k <= 0:
            return 0.0
        return 1.0 if block_map[k] == block_map[k - 1] else 0.0

    def _set_terminal_k_bounds(self, Kf: float):
        if self.cfg.enable_terminal_K_hard:
            lb = np.array([Kf], dtype=float)
            ub = np.array([Kf], dtype=float)
        else:
            lb = np.array([-self.cfg.K_MAX], dtype=float)
            ub = np.array([self.cfg.K_MAX], dtype=float)
        self._solver.constraints_set(self.cfg.N, "lbx", lb)
        self._solver.constraints_set(self.cfg.N, "ubx", ub)

    # -------------------------------------------------------------------------
    # model / solver build
    # -------------------------------------------------------------------------
    def _build_model(self) -> AcadosModel:
        model = AcadosModel()
        model.name = "rota_optimal_ds_acados"

        x = ca.SX.sym("x", self.NX_AUG)
        u = ca.SX.sym("u", self.NU)
        p = ca.SX.sym("p", self.NP)

        Xk = x[self.AIX_X]
        Yk = x[self.AIX_Y]
        psik = x[self.AIX_PSI]
        Kk = x[self.AIX_K]
        Kcmd_prev = x[self.AIX_KCMD_PREV]
        ds_prev = x[self.AIX_DS_PREV]

        Kcmd = u[self.IU_KCMD]
        ds = u[self.IU_DS]

        xg = p[0]
        yg = p[1]
        psig = p[2]
        Kf = p[3]

        xwp = p[4]
        ywp = p[5]
        w_wp = p[6]

        xhit = p[7]
        yhit = p[8]
        psihit = p[9]
        Khit = p[10]
        hit_scale = p[11]

        term_scale = p[12]
        x0_ref = p[13]
        y0_ref = p[14]
        smooth_scale = p[15]
        kcmd_hold_scale = p[16]
        ds_hold_scale = p[17]

        # discrete dynamics
        K1 = self.K_next_fixed_ramp(Kk, Kcmd, ds, self.cfg.K_MAX, self.cfg.S_MAX)
        X1, Y1, psi1 = self.clothoid_increment_numeric(
            Xk, Yk, psik, Kk, K1, ds, self.cfg.nseg
        )

        x_next = ca.vertcat(
            X1,
            Y1,
            psi1,
            K1,
            Kcmd,
            ds,
        )

        model.x = x
        model.u = u
        model.p = p
        model.disc_dyn_expr = x_next

        Dref2 = (xg - x0_ref) ** 2 + (yg - y0_ref) ** 2
        pos_scale = Dref2 + 1.0

        # stage cost
        stage_cost = 0.0
        stage_cost += self.cfg.w_K * (Kk ** 2)
        stage_cost += self.cfg.w_Kcmd * (Kcmd ** 2)
        stage_cost += smooth_scale * self.cfg.w_dKcmd * ((Kcmd - Kcmd_prev) ** 2)
        stage_cost += smooth_scale * self.cfg.w_ds_smooth * ((ds - ds_prev) ** 2)

        dwp2 = (X1 - xwp) ** 2 + (Y1 - ywp) ** 2
        stage_cost += w_wp * dwp2 / pos_scale

        psi_hit_e = self.wrap_to_pi(psi1 - psihit)
        dhit2 = (X1 - xhit) ** 2 + (Y1 - yhit) ** 2
        stage_cost += hit_scale * (self.cfg.w_pos * dhit2 / pos_scale)
        stage_cost += hit_scale * (self.cfg.w_psi * (psi_hit_e ** 2))
        stage_cost += hit_scale * (self.cfg.w_Kf * ((K1 - Khit) ** 2))

        model.cost_expr_ext_cost = stage_cost
        model.cost_expr_ext_cost_0 = stage_cost

        # terminal cost
        pos_e = (x[self.AIX_X] - xg) ** 2 + (x[self.AIX_Y] - yg) ** 2
        psi_e = self.wrap_to_pi(x[self.AIX_PSI] - psig)
        K_e = x[self.AIX_K] - Kf

        terminal_cost = 0.0
        terminal_cost += term_scale * (self.cfg.w_pos * pos_e / pos_scale)
        terminal_cost += term_scale * (self.cfg.w_psi * (psi_e ** 2))
        terminal_cost += term_scale * (self.cfg.w_Kf * (K_e ** 2))

        model.cost_expr_ext_cost_e = terminal_cost

        h_expr = [
            kcmd_hold_scale * (Kcmd - Kcmd_prev),
            ds_hold_scale * (ds - ds_prev),
        ]
        if self.cfg.ds_jump_max is not None and self.cfg.ds_jump_max > 0.0:
            h_expr.append(ds - ds_prev)
        model.con_h_expr = ca.vertcat(*h_expr)
        model.con_h_expr_0 = model.con_h_expr
        return model

    def _build_solver(self):
        model = self._build_model()

        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.cfg.N

        ocp.solver_options.tf = float(self.cfg.N)
        ocp.solver_options.integrator_type = "DISCRETE"

        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_0 = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"

        ocp.parameter_values = np.zeros((self.NP,), dtype=float)

        # intermediate state bounds: only K
        ocp.constraints.idxbx = np.array([self.IX_K], dtype=np.int64)
        ocp.constraints.lbx = np.array([-self.cfg.K_MAX], dtype=float)
        ocp.constraints.ubx = np.array([self.cfg.K_MAX], dtype=float)

        # terminal bounds: only K
        ocp.constraints.idxbx_e = np.array([self.IX_K], dtype=np.int64)
        ocp.constraints.lbx_e = np.array([-self.cfg.K_MAX], dtype=float)
        ocp.constraints.ubx_e = np.array([self.cfg.K_MAX], dtype=float)

        # initial state equality
        ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        ocp.constraints.lbx_0 = np.zeros((self.NX_AUG,), dtype=float)
        ocp.constraints.ubx_0 = np.zeros((self.NX_AUG,), dtype=float)

        # input bounds
        ocp.constraints.idxbu = np.array([self.IU_KCMD, self.IU_DS], dtype=np.int64)
        ocp.constraints.lbu = np.array(
            [-self.cfg.K_MAX, self.cfg.ds_min],
            dtype=float,
        )
        ocp.constraints.ubu = np.array(
            [self.cfg.K_MAX, self.cfg.ds_max],
            dtype=float,
        )

        h_lb = [0.0, 0.0]
        h_ub = [0.0, 0.0]
        if self.cfg.ds_jump_max is not None and self.cfg.ds_jump_max > 0.0:
            jmax = abs(self.cfg.ds_jump_max)
            h_lb.append(-jmax)
            h_ub.append(jmax)
        ocp.constraints.lh = np.array(h_lb, dtype=float)
        ocp.constraints.uh = np.array(h_ub, dtype=float)
        ocp.constraints.lh_0 = np.array(h_lb, dtype=float)
        ocp.constraints.uh_0 = np.array(h_ub, dtype=float)

        # solver options inspired by your older acados example
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.regularize_method = "CONVEXIFY"

        ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        ocp.solver_options.globalization_alpha_min = 0.05
        ocp.solver_options.globalization_alpha_reduction = 0.7
        ocp.solver_options.globalization_line_search_use_sufficient_descent = 1
        ocp.solver_options.globalization_use_SOC = 1

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.qp_solver_cond_N = min(10, self.cfg.N)
        ocp.solver_options.qp_solver_iter_max = 400

        ocp.solver_options.hpipm_mode = "ROBUST"
        ocp.solver_options.qp_solver_ric_alg = 0
        ocp.solver_options.qp_solver_cond_ric_alg = 0
        ocp.solver_options.qp_solver_warm_start = 1

        ocp.solver_options.nlp_solver_max_iter = 80
        ocp.solver_options.tol = 1e-6
        ocp.solver_options.print_level = 0

        code_export_dir = os.path.join(os.getcwd(), "c_generated_code_acados")
        if os.path.isdir(code_export_dir):
            shutil.rmtree(code_export_dir, ignore_errors=True)
        ocp.code_export_directory = code_export_dir

        solver = AcadosOcpSolver(ocp, json_file="acados_ocp_rota.json")

        self._ocp = ocp
        self._solver = solver

    # -------------------------------------------------------------------------
    # parameter packing
    # -------------------------------------------------------------------------
    def _make_stage_param(
        self,
        xg: float,
        yg: float,
        psig: float,
        Kf: float,
        xwp: float,
        ywp: float,
        w_wp: float,
        xhit: float,
        yhit: float,
        psihit: float,
        Khit: float,
        hit_scale: float,
        term_scale: float,
        x0_ref: float,
        y0_ref: float,
        smooth_scale: float,
        kcmd_hold_scale: float,
        ds_hold_scale: float,
    ) -> np.ndarray:
        return np.array(
            [
                xg, yg, psig, Kf,
                xwp, ywp, w_wp,
                xhit, yhit, psihit, Khit, hit_scale,
                term_scale,
                x0_ref, y0_ref,
                smooth_scale,
                kcmd_hold_scale, ds_hold_scale,
            ],
            dtype=float,
        )

    def _set_all_stage_params(
        self,
        x0: float,
        y0: float,
        xg: float,
        yg: float,
        psig: float,
        Kf: float,
        xwp: float,
        ywp: float,
        w_wp: float,
        xhit: float,
        yhit: float,
        psihit: float,
        Khit: float,
        hit_scale: float,
        term_scale: float,
    ):
        for k in range(self.cfg.N):
            hs = hit_scale if k == (self.k_hit - 1) else 0.0
            smooth_scale = 0.0 if k == 0 else 1.0
            kcmd_hold_scale = self._hold_scale(self.bl_kcmd, k)
            ds_hold_scale = self._hold_scale(self.bl_ds, k)

            p = self._make_stage_param(
                xg, yg, psig, Kf,
                xwp, ywp, w_wp,
                xhit, yhit, psihit, Khit, hs,
                0.0,
                x0, y0,
                smooth_scale,
                kcmd_hold_scale, ds_hold_scale,
            )
            self._solver.set(k, "p", p)

        pN = self._make_stage_param(
            xg, yg, psig, Kf,
            xwp, ywp, 0.0,
            xhit, yhit, psihit, Khit, 0.0,
            term_scale,
            x0, y0,
            0.0,
            0.0, 0.0,
        )
        self._solver.set(self.cfg.N, "p", pN)

    # -------------------------------------------------------------------------
    # warm start
    # -------------------------------------------------------------------------
    def warm_start(
        self,
        x0: float,
        y0: float,
        psi0: float,
        K0: float,
        kcmd_prev: float,
        ds_prev: float,
        xg: float,
        yg: float,
        psig: float,
        ds_seed: Optional[float],
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

        xk = self._pack_aug_state(x0, y0, psi0, K0, kcmd_prev, ds_prev)
        self._solver.set(0, "x", xk)

        for k in range(self.cfg.N):
            uk = np.array([Kcmd_ws[k], ds_ws[k]], dtype=float)
            self._solver.set(k, "u", uk)

            K1 = self.K_next_fixed_ramp_np(xk[self.AIX_K], Kcmd_ws[k], ds_ws[k], self.cfg.K_MAX, self.cfg.S_MAX)
            X1, Y1, PSI1 = self.clothoid_increment_numeric_np(
                xk[self.AIX_X], xk[self.AIX_Y], xk[self.AIX_PSI], xk[self.AIX_K], K1, ds_ws[k], self.cfg.nseg
            )
            xk = self._pack_aug_state(X1, Y1, PSI1, K1, Kcmd_ws[k], ds_ws[k])
            self._solver.set(k + 1, "x", xk)

    def apply_warm_start(
        self,
        x0: float,
        y0: float,
        psi0: float,
        K0: float,
        kcmd_prev: float,
        ds_prev: float,
        xg: float,
        yg: float,
        psig: float,
        use_last_warm: bool,
        ds_seed: Optional[float],
    ):
        x0_now = self._pack_aug_state(x0, y0, psi0, K0, kcmd_prev, ds_prev)

        if use_last_warm and self.last_warm.valid:
            self._solver.set(0, "x", x0_now)

            for k in range(1, self.cfg.N):
                xk = self._pack_aug_state(
                    self.last_warm.X[4 * k + 0],
                    self.last_warm.X[4 * k + 1],
                    self.last_warm.X[4 * k + 2],
                    self.last_warm.X[4 * k + 3],
                    self.last_warm.Kcmd[k - 1],
                    self.last_warm.ds[k - 1],
                )
                self._solver.set(k, "x", xk)

            xN = self._pack_aug_state(
                self.last_warm.X[4 * self.cfg.N + 0],
                self.last_warm.X[4 * self.cfg.N + 1],
                self.last_warm.X[4 * self.cfg.N + 2],
                self.last_warm.X[4 * self.cfg.N + 3],
                self.last_warm.Kcmd[self.cfg.N - 1],
                self.last_warm.ds[self.cfg.N - 1],
            )
            self._solver.set(self.cfg.N, "x", xN)

            for k in range(self.cfg.N):
                uk = np.array([self.last_warm.Kcmd[k], self.last_warm.ds[k]], dtype=float)
                self._solver.set(k, "u", uk)
            return

        self.warm_start(x0, y0, psi0, K0, kcmd_prev, ds_prev, xg, yg, psig, ds_seed)

    def shift_solution(self, X_sol: np.ndarray, U_sol: np.ndarray):
        ws = WarmStartData(
            X=[0.0] * ((self.cfg.N + 1) * 4),
            Kcmd=[0.0] * self.cfg.N,
            ds=[0.0] * self.cfg.N,
            valid=True,
        )

        for k in range(self.cfg.N):
            xsrc = X_sol[k + 1]
            for i in range(4):
                ws.X[k * 4 + i] = float(xsrc[i])

        for i in range(4):
            ws.X[self.cfg.N * 4 + i] = float(X_sol[self.cfg.N, i])

        for k in range(self.cfg.N - 1):
            ws.Kcmd[k] = float(U_sol[k + 1, self.IU_KCMD])
            ws.ds[k] = float(U_sol[k + 1, self.IU_DS])

        ws.Kcmd[self.cfg.N - 1] = float(U_sol[self.cfg.N - 1, self.IU_KCMD])
        ws.ds[self.cfg.N - 1] = float(U_sol[self.cfg.N - 1, self.IU_DS])

        self.last_warm = ws

    # -------------------------------------------------------------------------
    # solve
    # -------------------------------------------------------------------------
    def solve(
        self,
        x0: float,
        y0: float,
        psi0: float,
        K0: float,
        xg: float,
        yg: float,
        psig: float,
        Kf: float,
        term_scale: float = 1.0,
        w_wp: float = 0.0,
        xwp: Optional[float] = None,
        ywp: Optional[float] = None,
        hit_scale: float = 0.0,
        xhit: Optional[float] = None,
        yhit: Optional[float] = None,
        psihit: Optional[float] = None,
        Khit: Optional[float] = None,
        ds_prev: Optional[float] = None,
        ds_seed: Optional[float] = None,
        use_last_warm: bool = True,
    ) -> MPCSolution:
        xwp_v = xg if xwp is None else xwp
        ywp_v = yg if ywp is None else ywp
        xhit_v = xg if xhit is None else xhit
        yhit_v = yg if yhit is None else yhit
        psihit_v = psig if psihit is None else psihit
        Khit_v = Kf if Khit is None else Khit

        ds_prev_v = self.last_ds_applied if ds_prev is None else ds_prev
        kcmd_prev_v = self.last_kcmd_applied

        x0_aug = self._pack_aug_state(x0, y0, psi0, K0, kcmd_prev_v, ds_prev_v)

        def _prepare_and_initialize(use_warm: bool):
            self._set_all_stage_params(
                x0=x0,
                y0=y0,
                xg=xg,
                yg=yg,
                psig=psig,
                Kf=Kf,
                xwp=xwp_v,
                ywp=ywp_v,
                w_wp=w_wp,
                xhit=xhit_v,
                yhit=yhit_v,
                psihit=psihit_v,
                Khit=Khit_v,
                hit_scale=hit_scale,
                term_scale=term_scale,
            )

            self.apply_warm_start(
                x0, y0, psi0, K0,
                kcmd_prev_v, ds_prev_v,
                xg, yg, psig,
                use_warm, ds_seed
            )

            self._solver.constraints_set(0, "lbx", x0_aug)
            self._solver.constraints_set(0, "ubx", x0_aug)
            self._set_terminal_k_bounds(Kf)

        def _solve_once(use_warm: bool):
            _prepare_and_initialize(use_warm)

            try:
                init_res = self._solver.get_initial_residuals()
                if use_warm and init_res is not None and len(init_res) >= 2:
                    if float(init_res[0]) > 1e2 or float(init_res[1]) > 1.0:
                        self._solver.reset(reset_qp_solver_mem=1)
                        self.last_warm.valid = False
                        _prepare_and_initialize(False)
            except Exception:
                pass

            t0 = time.perf_counter()
            status_local = self._solver.solve()
            dt_local = time.perf_counter() - t0
            return status_local, dt_local

        status, dt = _solve_once(use_last_warm)
        self.last_solve_time_s = dt

        if status != 0:
            self._solver.print_statistics()

            if status == 2:
                try:
                    res = self._solver.get_residuals()
                    if res[0] < 2e-2 and res[1] < 1e-4:
                        print("acados reached max iterations but residuals are acceptable; accepting solution.")
                        status = 0
                except Exception:
                    pass

        if status == 4:
            print("acados hard QP failure detected; resetting solver and retrying with cold start.")
            self._solver.reset(reset_qp_solver_mem=1)
            self.last_warm.valid = False

            status, dt = _solve_once(False)
            self.last_solve_time_s = dt

            if status != 0:
                self._solver.print_statistics()

                if status == 2:
                    try:
                        res = self._solver.get_residuals()
                        if res[0] < 2e-2 and res[1] < 1e-4:
                            print("acados retry reached max iterations but residuals are acceptable; accepting solution.")
                            status = 0
                    except Exception:
                        pass

        if status != 0:
            raise RuntimeError(f"acados solve failed with status={status}")

        X_sol = np.zeros((self.cfg.N + 1, self.NX_AUG), dtype=float)
        U_sol = np.zeros((self.cfg.N, self.NU), dtype=float)

        for k in range(self.cfg.N + 1):
            X_sol[k, :] = self._solver.get(k, "x")
        for k in range(self.cfg.N):
            U_sol[k, :] = self._solver.get(k, "u")

        X4 = []
        for k in range(self.cfg.N + 1):
            X4.extend(
                [
                    float(X_sol[k, self.IX_X]),
                    float(X_sol[k, self.IX_Y]),
                    float(X_sol[k, self.IX_PSI]),
                    float(X_sol[k, self.IX_K]),
                ]
            )

        out = MPCSolution(
            X=X4,
            Kcmd=U_sol[:, self.IU_KCMD].astype(float).tolist(),
            ds=U_sol[:, self.IU_DS].astype(float).tolist(),
            start=State4(x0, y0, psi0, K0),
            goal=State4(xg, yg, psig, Kf),
            xwp=xwp_v,
            ywp=ywp_v,
            w_wp=w_wp,
            term_scale=term_scale,
        )

        self.last_sol = out
        self.shift_solution(X_sol, U_sol)

        self.last_kcmd_applied = out.Kcmd[0]
        self.last_ds_applied = out.ds[0]

        print(f"acados solve time: {dt:.4f} s")
        return out

    def mpc_step(
        self,
        state: State4,
        goal: State4,
        term_scale: float = 1.0,
        w_wp: float = 0.0,
        xwp: Optional[float] = None,
        ywp: Optional[float] = None,
        hit_scale: float = 0.0,
        xhit: Optional[float] = None,
        yhit: Optional[float] = None,
        psihit: Optional[float] = None,
        Khit: Optional[float] = None,
        ds_seed: Optional[float] = None,
    ) -> StepOutput:
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

        return StepOutput(
            state=State4(x1, y1, psi1, K1),
            ds0=ds0,
            Kcmd0=sol.Kcmd[0],
            sol=sol,
        )

    # -------------------------------------------------------------------------
    # log writer
    # -------------------------------------------------------------------------
    @staticmethod
    def write_log_csv(log: RecedingLog, out_path: str):
        with open(out_path, "w", newline="", encoding="utf-8") as f:
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
                    "solve_time_s",
                ]
            )

            n_steps = len(log.traj)
            for k in range(n_steps):
                x, y = log.traj[k]
                psi = log.psi[k]
                K = log.K[k]

                Kcmd = log.Kcmd[k - 1] if k > 0 and (k - 1) < len(log.Kcmd) else ""
                ds = log.ds[k - 1] if k > 0 and (k - 1) < len(log.ds) else ""
                wp_index = log.wp_index[k] if k < len(log.wp_index) else ""
                detour_wp_x = log.detour_wp_x[k] if k < len(log.detour_wp_x) else ""
                detour_wp_y = log.detour_wp_y[k] if k < len(log.detour_wp_y) else ""
                detour_kf = log.detour_kf[k] if k < len(log.detour_kf) else ""
                detour_obs_idx = log.detour_obs_idx[k] if k < len(log.detour_obs_idx) else ""
                solve_time_s = log.solve_time_s[k - 1] if k > 0 and (k - 1) < len(log.solve_time_s) else ""

                writer.writerow(
                    [
                        k,
                        x,
                        y,
                        psi,
                        K,
                        Kcmd,
                        ds,
                        wp_index,
                        detour_wp_x,
                        detour_wp_y,
                        detour_kf,
                        detour_obs_idx,
                        solve_time_s,
                    ]
                )

    # -------------------------------------------------------------------------
    # receding horizon multi-waypoint
    # -------------------------------------------------------------------------
    def run_receding_horizon_multi(
        self,
        waypoints: list[Waypoint],
        initial_state: State4,
        opts: Optional[RecedingOptions] = None,
    ) -> RecedingLog:
        from obstacle_avoidance import select_obstacle_detour_waypoint

        if not waypoints:
            raise RuntimeError("Waypoint list cannot be empty.")
        if opts is None:
            opts = RecedingOptions()

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
        self.last_kcmd_applied = 0.0

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

        for _it in range(opts.max_iters):
            wp_main = waypoints[cur_idx]

            if detour_wp is None and opts.enable_obstacle_avoidance and opts.obstacles:
                detour = select_obstacle_detour_waypoint(
                    x, y, psi, wp_main, opts.obstacles, obstacle_done_for_wp,
                    opts.obstacle_clearance, opts.obstacle_trigger_margin, opts.obstacle_waypoint_tol,
                )
                if detour is not None:
                    detour_wp = detour.waypoint
                    detour_obstacle_idx = detour.obstacle_index
                    self.last_warm.valid = False
                    ds_seed_next = log.ds[-1] if log.ds else None

            using_detour = detour_wp is not None
            wp = detour_wp if using_detour else wp_main
            Xf, Yf = wp.X, wp.Y
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

            ds_seed_next = None
            state = step.state
            x, y, psi, K = state.x, state.y, state.psi, state.K

            log.traj.append((x, y))
            log.psi.append(psi)
            log.K.append(K)
            log.Kcmd.append(step.Kcmd0)
            log.ds.append(step.ds0)
            log.solve_time_s.append(self.last_solve_time_s)
            log.wp_index.append(cur_idx)

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
