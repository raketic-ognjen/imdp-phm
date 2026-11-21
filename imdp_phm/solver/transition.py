"""Fast Gauss-Hermite Quadrature Transition Expectation Operator E_theta[V(H_{t+dt}, lam_{t+dt}) | h, lam, a]
and Baseline Point-Estimate MDP Solver (WP2 Step 3A).
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from imdp_phm.core.params import HawkesMertonParams, StructuralConstants
from imdp_phm.core.sde import CoupledHawkesMertonSDE
from imdp_phm.solver.grid import MaintenanceAction, MaintenanceCostModel, StateGrid2D


@dataclass
class PointMDPResult:
    """Solution of classical point-estimate Bellman equation for a fixed parameter vector theta."""
    value_function: np.ndarray     # (n_h, n_lam) V*(h, lambda)
    optimal_policy: np.ndarray     # (n_h, n_lam) int action map in {0, 1, 2}
    q_values: np.ndarray           # (n_h, n_lam, 3) Q*(h, lambda, a)
    iterations: int
    bellman_residual: float
    converged: bool

    def action_at(self, grid: StateGrid2D, h: float, lam: float) -> MaintenanceAction:
        q_interp = [float(grid.interpolate_bilinear(self.q_values[:, :, a], np.array([h]), np.array([lam]))[0]) for a in range(3)]
        return MaintenanceAction(int(np.argmin(q_interp)))

    def value_at(self, grid: StateGrid2D, h: float, lam: float) -> float:
        return float(grid.interpolate_bilinear(self.value_function, np.array([h]), np.array([lam]))[0])


class SingleStepQuadratureOperator:
    """Computes exact one-step transition expectations E_theta[V(H_{t+Delta t}, lambda_{t+Delta t}) | h, lam, a]
    across the 2D state grid using Gauss-Hermite quadrature.
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
        self._stage_costs = self.cost_model.stage_cost_grid(self.grid)

    def expected_next_value(
        self,
        v_table: np.ndarray,
        params: HawkesMertonParams,
        action: MaintenanceAction,
    ) -> np.ndarray:
        v = np.asarray(v_table, dtype=np.float64)
        n_h, n_lam = self.grid.shape
        if v.shape != (n_h, n_lam):
            raise ValueError(f"Expected value table shape {(n_h, n_lam)}, got {v.shape}")

        if action == MaintenanceAction.PREVENTIVE_REPLACE:
            v_reset = float(
                self.grid.interpolate_bilinear(
                    v,
                    np.array([self.grid.h0], dtype=np.float64),
                    np.array([params.mu0], dtype=np.float64),
                )[0]
            )
            return np.full((n_h, n_lam), v_reset, dtype=np.float64)

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

        lam_no_jump = params.mu0 + (lam_eff - params.mu0) * rho
        h_no_jump = np.maximum(
            0.0,
            self._h_mesh[:, :, None] + a_step[:, :, None] + sig_sqrt_dt[:, :, None] * self._zw[None, None, :],
        )
        lam_no_jump_3d = np.broadcast_to(lam_no_jump[:, :, None], h_no_jump.shape)
        v_no_jump_nodes = self.grid.interpolate_bilinear(v, h_no_jump, lam_no_jump_3d)
        ev_no_jump = np.tensordot(v_no_jump_nodes, self._ww, axes=([2], [0]))

        lam_jump = lam_no_jump + alpha_h
        y_nodes = np.exp(params.mu_J + params.sigma_J * self._zy)

        h_jump = h_no_jump[:, :, :, None] + y_nodes[None, None, None, :]
        lam_jump_4d = np.broadcast_to(lam_jump[:, :, None, None], h_jump.shape)
        v_jump_nodes = self.grid.interpolate_bilinear(v, h_jump, lam_jump_4d)
        ev_jump_w = np.tensordot(v_jump_nodes, self._wy, axes=([3], [0]))
        ev_jump = np.tensordot(ev_jump_w, self._ww, axes=([2], [0]))

        ev_total = (1.0 - p_jump) * ev_no_jump + p_jump * ev_jump
        failed = self._h_mesh >= self.grid.h_crit - 1e-12
        if np.any(failed):
            ev_total = np.where(failed, v, ev_total)

        return ev_total

    def all_action_expectations(
        self,
        v_table: np.ndarray,
        params: HawkesMertonParams,
    ) -> np.ndarray:
        e0 = self.expected_next_value(v_table, params, MaintenanceAction.CONTINUE)
        e1 = self.expected_next_value(v_table, params, MaintenanceAction.NDE_INSPECT)
        e2 = self.expected_next_value(v_table, params, MaintenanceAction.PREVENTIVE_REPLACE)
        return np.stack([e0, e1, e2], axis=-1)


QuadratureTransitionOperator = SingleStepQuadratureOperator


def solve_point_mdp(
    grid: StateGrid2D,
    params: HawkesMertonParams,
    cost_model: Optional[MaintenanceCostModel] = None,
    constants: Optional[StructuralConstants] = None,
    tol: float = 1e-6,
    max_iter: int = 2000,
    n_quad_w: int = 7,
    n_quad_y: int = 9,
) -> PointMDPResult:
    cm = cost_model or MaintenanceCostModel()
    consts = constants or StructuralConstants()
    n_h, n_lam = grid.shape
    n_states = n_h * n_lam

    from scipy.sparse import eye as speye, vstack as spvstack
    from scipy.sparse.linalg import spsolve
    from imdp_phm.solver.bellman import SparseTransitionCompiler

    compiler = SparseTransitionCompiler(
        grid=grid,
        cost_model=cm,
        constants=consts,
        n_quad_w=n_quad_w,
        n_quad_y=n_quad_y,
    )
    p0, p1, p2 = compiler.all_action_matrices(params)
    p_mats = (p0, p1, p2)

    stage_costs = cm.stage_cost_grid(grid)
    stage_costs_flat = stage_costs.reshape(n_states, 3)
    gamma_d = cm.discount_factor

    v_flat = np.min(stage_costs_flat, axis=-1).copy()
    residual = np.inf
    converged = False
    n_iter = 0

    for it in range(1, max_iter + 1):
        n_iter = it
        q_flat = np.column_stack([
            stage_costs_flat[:, 0] + gamma_d * p0.dot(v_flat),
            stage_costs_flat[:, 1] + gamma_d * p1.dot(v_flat),
            stage_costs_flat[:, 2] + gamma_d * p2.dot(v_flat),
        ])
        v_next = np.min(q_flat, axis=-1)
        policy = np.argmin(q_flat, axis=-1).astype(np.int32)
        residual = float(np.max(np.abs(v_next - v_flat)))
        if residual < tol:
            v_flat = v_next
            converged = True
            break

        c_pol = stage_costs_flat[np.arange(n_states), policy]
        row_slices = [p_mats[policy[s]].getrow(s) for s in range(n_states)]
        p_pol = spvstack(row_slices, format="csr")
        lhs = speye(n_states, format="csc") - gamma_d * p_pol.tocsc()
        v_flat = np.asarray(spsolve(lhs, c_pol), dtype=np.float64)

    q_final = np.column_stack([
        stage_costs_flat[:, 0] + gamma_d * p0.dot(v_flat),
        stage_costs_flat[:, 1] + gamma_d * p1.dot(v_flat),
        stage_costs_flat[:, 2] + gamma_d * p2.dot(v_flat),
    ]).reshape(n_h, n_lam, 3)
    v_final = np.min(q_final, axis=-1)
    policy_final = np.argmin(q_final, axis=-1).astype(np.int32)

    return PointMDPResult(
        value_function=v_final,
        optimal_policy=policy_final,
        q_values=q_final,
        iterations=n_iter,
        bellman_residual=residual,
        converged=converged,
    )
