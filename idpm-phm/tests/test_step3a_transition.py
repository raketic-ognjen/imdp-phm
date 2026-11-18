"""Unit and numerical verification tests for Step 3A (WP2 2D Grid, Quadrature Operator & Point MDP)."""

import numpy as np
from imdp_phm.core import CoupledHawkesMertonSDE, HawkesMertonParams, StructuralConstants
from imdp_phm.solver import (
    MaintenanceAction,
    MaintenanceCostModel,
    StateGrid2D,
    SingleStepQuadratureOperator,
    solve_point_mdp,
)


def test_quadrature_expectation_matches_monte_carlo():
    """Verify Gauss-Hermite quadrature transition operator matches 120,000-path single-step Monte Carlo."""
    params = HawkesMertonParams(a0=0.015, sigma0=0.025, mu0=0.40, alpha0=0.45, beta=0.90, gamma=1.40, mu_J=-2.2, sigma_J=0.55)
    consts = StructuralConstants(h0=0.10, h_crit=2.50, dt=0.1667)
    grid = StateGrid2D.uniform(n_h=51, n_lam=31, lam_min=0.05, lam_max=2.50, constants=consts)

    h_mesh, lam_mesh = grid.meshgrid()
    v_test = 100.0 * ((h_mesh / consts.h_crit) ** 2) + 40.0 * lam_mesh

    op = SingleStepQuadratureOperator(grid=grid, constants=consts, n_quad_w=9, n_quad_y=11)
    ev_quad = op.expected_next_value(v_test, params, MaintenanceAction.CONTINUE)

    ih, il = 20, 10
    h_start = float(grid.h_grid[ih])
    lam_start = float(grid.lam_grid[il])

    sde = CoupledHawkesMertonSDE(params=params, constants=consts)
    mc = sde.simulate_macro_epochs(n_paths=120000, t_max=consts.dt, seed=2026, h_init=h_start, lam_init=lam_start)
    v_mc_samples = grid.interpolate_bilinear(v_test, mc.h_paths[:, 1], mc.lambda_paths[:, 1])
    ev_mc = float(np.mean(v_mc_samples))

    assert np.isclose(ev_quad[ih, il], ev_mc, rtol=0.01), f"Quad {ev_quad[ih, il]:.4f} vs MC {ev_mc:.4f}"


def test_preventive_replacement_resets_to_pristine_state():
    """Verify PREVENTIVE_REPLACE produces constant expected next value equal to V(h0, mu0)."""
    params = HawkesMertonParams()
    consts = StructuralConstants()
    grid = StateGrid2D.uniform(n_h=31, n_lam=21, constants=consts)
    h_mesh, lam_mesh = grid.meshgrid()
    v_test = 50.0 * h_mesh + 10.0 * lam_mesh

    op = SingleStepQuadratureOperator(grid=grid, constants=consts)
    ev_rep = op.expected_next_value(v_test, params, MaintenanceAction.PREVENTIVE_REPLACE)
    expected_reset = 50.0 * consts.h0 + 10.0 * params.mu0
    assert np.allclose(ev_rep, expected_reset, atol=1e-10)


def test_point_mdp_convergence_and_monotonic_structure():
    """Verify point-estimate Bellman value iteration converges and yields physically ordered actions:
    CONTINUE at pristine states, PREVENTIVE_REPLACE near absorbing failure barrier H_crit.
    """
    params = HawkesMertonParams(a0=2.0e-3, sigma0=2.0e-3, mu0=0.25, alpha0=0.45, beta=0.85, gamma=1.40, mu_J=-2.2, sigma_J=0.6)
    consts = StructuralConstants(h0=0.10, h_crit=2.50, dt=0.1667)
    grid = StateGrid2D.uniform(n_h=41, n_lam=25, lam_min=0.10, lam_max=2.50, constants=consts)
    cm = MaintenanceCostModel(c_inspection=15.0, c_preventive=200.0, c_failure=3500.0, discount_factor=0.99)

    res = solve_point_mdp(grid=grid, params=params, cost_model=cm, constants=consts, tol=1e-5, max_iter=1500)
    assert res.converged is True
    assert res.bellman_residual < 1e-5

    diffs_h = np.diff(res.value_function, axis=0)
    assert np.all(diffs_h >= -1e-6)

    assert res.optimal_policy[2, 1] == int(MaintenanceAction.CONTINUE)
    assert res.optimal_policy[-1, 1] == int(MaintenanceAction.PREVENTIVE_REPLACE)
    assert res.optimal_policy[-2, -2] == int(MaintenanceAction.PREVENTIVE_REPLACE)
