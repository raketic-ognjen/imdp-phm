"""Minimax Bellman Optimality Solver for Imprecise Markov Decision Processes (WP2 Step 3C).

Solves:
    V*(h, lambda) = min_{a in A} { c(h, lambda, a) + gamma_d * T_bar_{Delta t}^a V*(h, lambda) }
where:
    T_bar_{Delta t}^a V(h, lambda) = sup_{theta in Theta} E_theta[V(H_{t+Delta t}, lambda_{t+Delta t}) | h, lam, a]
using precompiled sparse transition matrices P_theta^a, GP surrogate emulation with interior
L-BFGS-B optimization, and Howard / Modified Policy Iteration (Concept Note Eqs. 4-5 & Objectives 2-3).
"""

from dataclasses import dataclass
import time
from typing import Optional
import numpy as np
from scipy.sparse import csr_matrix, eye as speye, vstack as spvstack
from scipy.sparse.linalg import spsolve

from imdp_phm.core.params import HawkesMertonParams, StructuralConstants
from imdp_phm.core.sde import CoupledHawkesMertonSDE
from imdp_phm.inference.credal import CredalParameterSet
from imdp_phm.inference.mle import MacroEpochDataset, MLEEstimationResult, fit_hawkes_merton_mle
from imdp_phm.solver.grid import MaintenanceAction, MaintenanceCostModel, StateGrid2D
from imdp_phm.solver.surrogate import GaussianProcessUpperOperator
from imdp_phm.solver.transition import SingleStepQuadratureOperator


class SparseTransitionCompiler:
    """Compiles Gauss-Hermite quadrature transition probabilities for a fixed (theta, action) pair
    into a row-stochastic CSR sparse matrix P_theta^a in R^{(n_h * n_lam) x (n_h * n_lam)},
    reducing repeated expectation evaluations E_theta[V | h, lam, a] to microsecond sparse matvecs.
    """

    def __init__(
        self,
        grid: StateGrid2D,
        cost_model: Optional[MaintenanceCostModel] = None,
        constants: Optional[StructuralConstants] = None,
        n_quad_w: int = 7,
        n_quad_y: int = 9,
    ) -> None:
        self.grid = grid
        self.cost_model = cost_model or MaintenanceCostModel()
        self.constants = constants or StructuralConstants()
        self.n_quad_w = int(n_quad_w)
        self.n_quad_y = int(n_quad_y)

        xw, ww = np.polynomial.hermite.hermgauss(self.n_quad_w)
        self._zw = np.sqrt(2.0) * xw
        self._ww = ww / np.sqrt(np.pi)

        xy, wy = np.polynomial.hermite.hermgauss(self.n_quad_y)
        self._zy = np.sqrt(2.0) * xy
        self._wy = wy / np.sqrt(np.pi)

        self._h_mesh, self._lam_mesh = self.grid.meshgrid()
        self._cache: dict[tuple[bytes, int], csr_matrix] = {}

    def _bilinear_corners(
        self,
        hq: np.ndarray,
        lq: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h_c = np.clip(hq, self.grid.h_grid[0], self.grid.h_grid[-1])
        l_c = np.clip(lq, self.grid.lam_grid[0], self.grid.lam_grid[-1])

        ih = np.searchsorted(self.grid.h_grid, h_c, side="right") - 1
        ih = np.clip(ih, 0, len(self.grid.h_grid) - 2)
        wh = (h_c - self.grid.h_grid[ih]) / (self.grid.h_grid[ih + 1] - self.grid.h_grid[ih])

        il = np.searchsorted(self.grid.lam_grid, l_c, side="right") - 1
        il = np.clip(il, 0, len(self.grid.lam_grid) - 2)
        wl = (l_c - self.grid.lam_grid[il]) / (self.grid.lam_grid[il + 1] - self.grid.lam_grid[il])

        return ih, il, wh, wl, 1.0 - wh, 1.0 - wl

    def compile_matrix(
        self,
        params: HawkesMertonParams,
        action: MaintenanceAction,
    ) -> csr_matrix:
        key = (np.round(params.to_vector(), decimals=10).tobytes(), int(action))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        n_h, n_lam = self.grid.shape
        n_states = n_h * n_lam

        if action == MaintenanceAction.PREVENTIVE_REPLACE:
            ih, il, wh, wl, om_wh, om_wl = self._bilinear_corners(
                np.array([self.grid.h0], dtype=np.float64),
                np.array([params.mu0], dtype=np.float64),
            )
            cols = np.array([
                ih[0] * n_lam + il[0],
                (ih[0] + 1) * n_lam + il[0],
                ih[0] * n_lam + (il[0] + 1),
                (ih[0] + 1) * n_lam + (il[0] + 1),
            ], dtype=np.int32)
            wts = np.array([
                om_wh[0] * om_wl[0],
                wh[0] * om_wl[0],
                om_wh[0] * wl[0],
                wh[0] * wl[0],
            ], dtype=np.float64)

            row_idx = np.repeat(np.arange(n_states, dtype=np.int32), 4)
            col_idx = np.tile(cols, n_states)
            data = np.tile(wts, n_states)
            mat = csr_matrix((data, (row_idx, col_idx)), shape=(n_states, n_states))
            mat.sum_duplicates()
            self._cache[key] = mat
            return mat

        sde = CoupledHawkesMertonSDE(params=params, constants=self.constants)
        dt = self.constants.dt
        rho = float(np.exp(-params.beta * dt))

        if action == MaintenanceAction.NDE_INSPECT:
            relief = self.cost_model.inspection_hazard_relief
            lam_eff = params.mu0 + (1.0 - relief) * np.maximum(0.0, self._lam_mesh - params.mu0)
        else:
            lam_eff = self._lam_mesh

        p_jump = 1.0 - np.exp(-lam_eff * dt)
        a_step = sde.drift(self._h_mesh) * dt
        sig_sqrt_dt = sde.volatility(self._h_mesh) * np.sqrt(dt)
        alpha_h = sde.excitation_alpha(self._h_mesh)

        failed_2d = self._h_mesh >= self.grid.h_crit - 1e-12
        active_2d = ~failed_2d
        act_rows = np.flatnonzero(active_2d.ravel()).astype(np.int32)

        row_chunks = []
        col_chunks = []
        val_chunks = []

        failed_rows = np.flatnonzero(failed_2d.ravel()).astype(np.int32)
        if len(failed_rows) > 0:
            row_chunks.append(failed_rows)
            col_chunks.append(failed_rows)
            val_chunks.append(np.ones(len(failed_rows), dtype=np.float64))

        lam_no_jump = params.mu0 + (lam_eff - params.mu0) * rho
        h_nj = np.maximum(
            0.0,
            self._h_mesh[active_2d, None] + a_step[active_2d, None] + sig_sqrt_dt[active_2d, None] * self._zw[None, :],
        )
        l_nj = np.broadcast_to(lam_no_jump[active_2d, None], h_nj.shape)
        w_nj = (1.0 - p_jump[active_2d, None]) * self._ww[None, :]

        ih0, il0, wh0, wl0, om_wh0, om_wl0 = self._bilinear_corners(h_nj, l_nj)
        r_nj = np.broadcast_to(act_rows[:, None], h_nj.shape).ravel()
        for c_h, c_l, w_bilin in (
            (ih0, il0, om_wh0 * om_wl0),
            (ih0 + 1, il0, wh0 * om_wl0),
            (ih0, il0 + 1, om_wh0 * wl0),
            (ih0 + 1, il0 + 1, wh0 * wl0),
        ):
            row_chunks.append(r_nj)
            col_chunks.append((c_h * n_lam + c_l).ravel().astype(np.int32))
            val_chunks.append((w_nj * w_bilin).ravel())

        lam_jump = lam_no_jump + alpha_h
        y_nodes = np.exp(params.mu_J + params.sigma_J * self._zy)
        h_j = h_nj[:, :, None] + y_nodes[None, None, :]
        l_j = np.broadcast_to(lam_jump[active_2d, None, None], h_j.shape)
        w_j = p_jump[active_2d, None, None] * self._ww[None, :, None] * self._wy[None, None, :]

        ih1, il1, wh1, wl1, om_wh1, om_wl1 = self._bilinear_corners(h_j, l_j)
        r_j = np.broadcast_to(act_rows[:, None, None], h_j.shape).ravel()
        for c_h, c_l, w_bilin in (
            (ih1, il1, om_wh1 * om_wl1),
            (ih1 + 1, il1, wh1 * om_wl1),
            (ih1, il1 + 1, om_wh1 * wl1),
            (ih1 + 1, il1 + 1, wh1 * wl1),
        ):
            row_chunks.append(r_j)
            col_chunks.append((c_h * n_lam + c_l).ravel().astype(np.int32))
            val_chunks.append((w_j * w_bilin).ravel())

        mat = csr_matrix(
            (np.concatenate(val_chunks), (np.concatenate(row_chunks), np.concatenate(col_chunks))),
            shape=(n_states, n_states),
        )
        mat.sum_duplicates()
        self._cache[key] = mat
        return mat

    def all_action_matrices(self, params: HawkesMertonParams) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
        return (
            self.compile_matrix(params, MaintenanceAction.CONTINUE),
            self.compile_matrix(params, MaintenanceAction.NDE_INSPECT),
            self.compile_matrix(params, MaintenanceAction.PREVENTIVE_REPLACE),
        )


@dataclass
class IMDPBellmanResult:
    """Complete solution of the Minimax Imprecise Bellman Optimality Equation (WP2)."""
    value_function: np.ndarray            # (n_h, n_lam) V*(h, lambda)
    optimal_policy: np.ndarray            # (n_h, n_lam) a*(h, lambda) in {0, 1, 2}
    q_values: np.ndarray                  # (n_h, n_lam, 3) Q*(h, lambda, a)
    worst_case_theta: np.ndarray          # (n_h, n_lam, 3, 8) maximizing theta*(h, lambda, a) in Theta
    active_worst_case_theta: np.ndarray   # (n_h, n_lam, 8) theta*(h, lambda, a*(h, lambda))
    outer_iterations: int
    total_inner_steps: int
    bellman_residual: float
    converged: bool
    gp_r2_score: float
    solve_time_sec: float

    def action_at(self, grid: StateGrid2D, h: float, lam: float) -> MaintenanceAction:
        q_interp = [
            float(
                grid.interpolate_bilinear(
                    self.q_values[:, :, a],
                    np.array([h], dtype=np.float64),
                    np.array([lam], dtype=np.float64),
                )[0]
            )
            for a in range(3)
        ]
        return MaintenanceAction(int(np.argmin(q_interp)))

    def value_at(self, grid: StateGrid2D, h: float, lam: float) -> float:
        return float(
            grid.interpolate_bilinear(
                self.value_function,
                np.array([h], dtype=np.float64),
                np.array([lam], dtype=np.float64),
            )[0]
        )

    def policy_region_fractions(self) -> dict[str, float]:
        total = float(self.optimal_policy.size)
        return {
            "CONTINUE": float(np.sum(self.optimal_policy == int(MaintenanceAction.CONTINUE))) / total,
            "NDE_INSPECT": float(np.sum(self.optimal_policy == int(MaintenanceAction.NDE_INSPECT))) / total,
            "PREVENTIVE_REPLACE": float(np.sum(self.optimal_policy == int(MaintenanceAction.PREVENTIVE_REPLACE))) / total,
        }


def solve_imdp_minimax(
    grid: StateGrid2D,
    credal_set: CredalParameterSet,
    cost_model: Optional[MaintenanceCostModel] = None,
    constants: Optional[StructuralConstants] = None,
    tol: float = 1e-5,
    max_outer_iter: int = 40,
    inner_eval_steps: int = 80,
    n_design: int = 32,
    n_quad_w: int = 7,
    n_quad_y: int = 9,
    seed: int = 42,
) -> IMDPBellmanResult:
    """Solve the Imprecise Markov Decision Process minimax Bellman equation via GP-assisted
    Howard Policy Iteration with precompiled sparse transition matrices:
        V*(h, lam) = min_{a in A} { c(h, lam, a) + gamma_d * sup_{theta in Theta} E_theta[V* | h, lam, a] }.
    """
    t0 = time.perf_counter()
    cm = cost_model or MaintenanceCostModel()
    consts = constants or StructuralConstants()
    n_h, n_lam = grid.shape
    n_states = n_h * n_lam

    quad_op = SingleStepQuadratureOperator(
        grid=grid,
        cost_model=cm,
        constants=consts,
        n_quad_w=n_quad_w,
        n_quad_y=n_quad_y,
    )
    gp_op = GaussianProcessUpperOperator(
        quad_operator=quad_op,
        credal_set=credal_set,
        n_design=n_design,
        seed=seed,
    )
    compiler = SparseTransitionCompiler(
        grid=grid,
        cost_model=cm,
        constants=consts,
        n_quad_w=n_quad_w,
        n_quad_y=n_quad_y,
    )

    design_thetas = gp_op._design_theta
    design_mats = [
        compiler.all_action_matrices(HawkesMertonParams.from_vector(th))
        for th in design_thetas
    ]

    stage_costs = quad_op._stage_costs
    stage_costs_flat = stage_costs.reshape(n_states, 3)
    gamma_d = cm.discount_factor

    archived_thetas: list[np.ndarray] = []
    archived_mats: list[tuple[csr_matrix, csr_matrix, csr_matrix]] = []
    archived_keys: set[bytes] = {np.round(th, decimals=8).tobytes() for th in design_thetas}

    def _register_thetas(theta_candidates: np.ndarray) -> None:
        for th in theta_candidates:
            th_proj = credal_set.project(th)
            k_bytes = np.round(th_proj, decimals=8).tobytes()
            if k_bytes not in archived_keys:
                archived_keys.add(k_bytes)
                archived_thetas.append(th_proj)
                archived_mats.append(compiler.all_action_matrices(HawkesMertonParams.from_vector(th_proj)))

    def _sparse_eval_batch(v_tab: np.ndarray, theta_batch: np.ndarray) -> np.ndarray:
        vf = v_tab.ravel()
        out_arr = np.empty((len(theta_batch), n_states, 3), dtype=np.float64)
        for idx_b, th_b in enumerate(theta_batch):
            p0, p1, p2 = compiler.all_action_matrices(HawkesMertonParams.from_vector(th_b))
            out_arr[idx_b, :, 0] = p0.dot(vf)
            out_arr[idx_b, :, 1] = p1.dot(vf)
            out_arr[idx_b, :, 2] = p2.dot(vf)
        return out_arr.reshape(len(theta_batch), n_h, n_lam, 3)

    v = np.min(stage_costs, axis=-1).copy()
    bellman_residual = np.inf
    converged = False
    outer_iter = 0
    total_inner = 0

    for k in range(1, max_outer_iter + 1):
        outer_iter = k
        v_flat = v.ravel()

        m_des = len(design_thetas)
        precomp_design = np.empty((m_des, n_states, 3), dtype=np.float64)
        for m_idx, (p0, p1, p2) in enumerate(design_mats):
            precomp_design[m_idx, :, 0] = p0.dot(v_flat)
            precomp_design[m_idx, :, 1] = p1.dot(v_flat)
            precomp_design[m_idx, :, 2] = p2.dot(v_flat)

        if len(archived_thetas) > 0:
            m_arch = len(archived_thetas)
            extra_th_arr = np.asarray(archived_thetas, dtype=np.float64)
            extra_val_arr = np.empty((m_arch, n_states, 3), dtype=np.float64)
            for a_idx, (p0, p1, p2) in enumerate(archived_mats):
                extra_val_arr[a_idx, :, 0] = p0.dot(v_flat)
                extra_val_arr[a_idx, :, 1] = p1.dot(v_flat)
                extra_val_arr[a_idx, :, 2] = p2.dot(v_flat)
            extra_val_4d = extra_val_arr.reshape(m_arch, n_h, n_lam, 3)
        else:
            extra_th_arr = None
            extra_val_4d = None

        up_res = gp_op.upper_transition_surrogate(
            v,
            verify_exact_candidates=True,
            precomputed_design_vals=precomp_design.reshape(m_des, n_h, n_lam, 3),
            extra_eval_thetas=extra_th_arr,
            extra_eval_vals=extra_val_4d,
            batch_eval_fn=_sparse_eval_batch,
        )

        uniq_new = np.unique(np.round(up_res.worst_case_theta.reshape(-1, 8), decimals=8), axis=0)
        _register_thetas(uniq_new)

        q_bellman = stage_costs + gamma_d * up_res.upper_expectation
        v_bellman = np.min(q_bellman, axis=-1)
        policy_k = np.argmin(q_bellman, axis=-1).astype(np.int32)

        bellman_residual = float(np.max(np.abs(v_bellman - v)))
        if bellman_residual < tol:
            v = v_bellman
            converged = True
            break

        pol_flat = policy_k.ravel()
        c_pol = stage_costs_flat[np.arange(n_states), pol_flat]
        w_theta_all = up_res.worst_case_theta.reshape(n_states, 3, 8)
        w_theta_pol = w_theta_all[np.arange(n_states), pol_flat]

        keys_arr = np.hstack([np.round(w_theta_pol, decimals=8), pol_flat[:, None].astype(np.float64)])
        uniq_pairs, inv_pair = np.unique(keys_arr, axis=0, return_inverse=True)
        pair_mats = [
            compiler.compile_matrix(HawkesMertonParams.from_vector(row[:8]), MaintenanceAction(int(row[8])))
            for row in uniq_pairs
        ]

        row_slices = [pair_mats[inv_pair[s]].getrow(s) for s in range(n_states)]
        p_pol = spvstack(row_slices, format="csr")
        total_inner += 1

        lhs_mat = speye(n_states, format="csc") - gamma_d * p_pol.tocsc()
        u = spsolve(lhs_mat, c_pol)
        v = np.asarray(u, dtype=np.float64).reshape(n_h, n_lam)

    v_flat = v.ravel()
    m_des = len(design_thetas)
    precomp_design = np.empty((m_des, n_states, 3), dtype=np.float64)
    for m_idx, (p0, p1, p2) in enumerate(design_mats):
        precomp_design[m_idx, :, 0] = p0.dot(v_flat)
        precomp_design[m_idx, :, 1] = p1.dot(v_flat)
        precomp_design[m_idx, :, 2] = p2.dot(v_flat)

    if len(archived_thetas) > 0:
        m_arch = len(archived_thetas)
        extra_th_arr = np.asarray(archived_thetas, dtype=np.float64)
        extra_val_arr = np.empty((m_arch, n_states, 3), dtype=np.float64)
        for a_idx, (p0, p1, p2) in enumerate(archived_mats):
            extra_val_arr[a_idx, :, 0] = p0.dot(v_flat)
            extra_val_arr[a_idx, :, 1] = p1.dot(v_flat)
            extra_val_arr[a_idx, :, 2] = p2.dot(v_flat)
        extra_val_4d = extra_val_arr.reshape(m_arch, n_h, n_lam, 3)
    else:
        extra_th_arr = None
        extra_val_4d = None

    final_up = gp_op.upper_transition_surrogate(
        v,
        verify_exact_candidates=True,
        precomputed_design_vals=precomp_design.reshape(m_des, n_h, n_lam, 3),
        extra_eval_thetas=extra_th_arr,
        extra_eval_vals=extra_val_4d,
        batch_eval_fn=_sparse_eval_batch,
    )
    q_final = stage_costs + gamma_d * final_up.upper_expectation
    v_final = np.min(q_final, axis=-1)
    policy_final = np.argmin(q_final, axis=-1).astype(np.int32)
    final_residual = float(np.max(np.abs(v_final - v)))
    if final_residual < tol:
        converged = True

    active_theta = np.take_along_axis(
        final_up.worst_case_theta,
        policy_final[:, :, None, None],
        axis=2,
    )[:, :, 0, :]

    elapsed = float(time.perf_counter() - t0)
    return IMDPBellmanResult(
        value_function=v_final,
        optimal_policy=policy_final,
        q_values=q_final,
        worst_case_theta=final_up.worst_case_theta,
        active_worst_case_theta=active_theta,
        outer_iterations=outer_iter,
        total_inner_steps=total_inner,
        bellman_residual=min(bellman_residual, final_residual),
        converged=converged,
        gp_r2_score=final_up.gp_r2_score,
        solve_time_sec=elapsed,
    )


def run_wp1_wp2_pipeline(
    dataset: MacroEpochDataset,
    grid: Optional[StateGrid2D] = None,
    cost_model: Optional[MaintenanceCostModel] = None,
    constants: Optional[StructuralConstants] = None,
    z_mult: float = 1.96,
    tol: float = 1e-5,
    seed: int = 42,
) -> tuple[MLEEstimationResult, CredalParameterSet, IMDPBellmanResult]:
    """End-to-end WP1 -> WP2 pipeline:
    1. Fit decoupled MLE + test H0 (NHPP) vs H1 (NH-Hawkes) with automatic NHPP fallback.
    2. Construct convex credal parameter set Theta subset R^8 with epistemic scarcity inflation.
    3. Solve minimax Bellman optimality equation for robust maintenance policy a*(h, lambda).
    """
    consts = constants or StructuralConstants()
    g = grid or StateGrid2D.uniform(n_h=41, n_lam=25, lam_min=0.05, lam_max=2.50, constants=consts)
    cm = cost_model or MaintenanceCostModel()

    mle_res = fit_hawkes_merton_mle(dataset, constants=consts)
    credal = CredalParameterSet.from_mle_result(
        mle_res,
        n_trajectories=dataset.h_paths.shape[0],
        z_mult=z_mult,
    )
    imdp_res = solve_imdp_minimax(
        grid=g,
        credal_set=credal,
        cost_model=cm,
        constants=consts,
        tol=tol,
        seed=seed,
    )
    return mle_res, credal, imdp_res
