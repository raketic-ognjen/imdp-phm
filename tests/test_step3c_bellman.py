"""Unit, mathematical contraction, and end-to-end WP1->WP2 verification tests for Step 3C (bellman.py)."""

import numpy as np
from imdp_phm.core import CoupledHawkesMertonSDE, HawkesMertonParams, StructuralConstants
from imdp_phm.inference import CredalParameterSet, MacroEpochDataset
from imdp_phm.solver import (
    MaintenanceAction,
    MaintenanceCostModel,
    StateGrid2D,
    SingleStepQuadratureOperator,
    SparseTransitionCompiler,
    solve_point_mdp,
    solve_imdp_minimax,
    run_wp1_wp2_pipeline,
)


def test_sparse_transition_matrix_matches_quadrature_and_is_row_stochastic():
    """Verify compiled CSR matrix P_theta^a is row-stochastic (rows sum to 1) and reproduces
    SingleStepQuadratureOperator expectations to machine precision across all 3 actions.
    """
    params = HawkesMertonParams(a0=0.008, sigma0=0.015, mu0=0.35, alpha0=0.45, beta=0.90, gamma=1.40, mu_J=-2.5, sigma_J=0.65)
    consts = StructuralConstants(h0=0.10, h_crit=2.50, dt=0.1667)
    grid = StateGrid2D.uniform(n_h=31, n_lam=21, lam_min=0.05, lam_max=2.50, constants=consts)
    cm = MaintenanceCostModel()

    op = SingleStepQuadratureOperator(grid=grid, cost_model=cm, constants=consts)
    compiler = SparseTransitionCompiler(grid=grid, cost_model=cm, constants=consts)

    h_mesh, lam_mesh = grid.meshgrid()
    v_test = 180.0 * ((h_mesh / consts.h_crit) ** 1.7) + 32.0 * lam_mesh

    for action in MaintenanceAction:
        p_mat = compiler.compile_matrix(params, action)
        row_sums = np.asarray(p_mat.sum(axis=1)).ravel()
        assert np.allclose(row_sums, 1.0, atol=1e-12), f"Row sums deviate from 1 for {action}"
        assert p_mat.data.min() >= -1e-15

        ev_sparse = p_mat.dot(v_test.ravel()).reshape(grid.shape)
        ev_quad = op.expected_next_value(v_test, params, action)
        assert np.allclose(ev_sparse, ev_quad, atol=1e-11), f"Mismatch between sparse P@v and quadrature for {action}"


def test_imdp_minimax_bellman_convergence_and_robust_dominance():
    """Verify solve_imdp_minimax converges rapidly (< 10s), satisfies Bellman fixed-point residual < tol,
    and yields upper value function V_IMDP*(h, lam) >= V_point*(h, lam) for every interior theta in Theta.
    """
    consts = StructuralConstants(h0=0.10, h_crit=2.50, dt=0.1667)
    grid = StateGrid2D.uniform(n_h=35, n_lam=21, lam_min=0.10, lam_max=2.50, constants=consts)
    cm = MaintenanceCostModel(c_inspection=18.0, c_preventive=220.0, c_failure=3500.0, discount_factor=0.99)

    nominal = HawkesMertonParams(a0=2.0e-3, sigma0=2.0e-3, mu0=0.25, alpha0=0.40, beta=0.90, gamma=1.30, mu_J=-2.3, sigma_J=0.60)
    base = nominal.to_vector()
    se = np.array([0.4e-3, 0.4e-3, 0.05, 0.08, 0.10, 0.25, 0.25, 0.12])
    credal = CredalParameterSet(lower=base - se, upper=base + se, max_branching_cap=0.99)

    imdp_res = solve_imdp_minimax(
        grid=grid,
        credal_set=credal,
        cost_model=cm,
        constants=consts,
        tol=1e-5,
        max_outer_iter=40,
        seed=42,
    )

    assert imdp_res.converged is True
    assert imdp_res.bellman_residual < 1e-5
    assert imdp_res.solve_time_sec < 10.0

    point_nom = solve_point_mdp(grid=grid, params=nominal, cost_model=cm, constants=consts, tol=1e-6)
    assert np.all(imdp_res.value_function + 1e-3 >= point_nom.value_function)

    lhs_samples = credal.sample_latin_hypercube(n_samples=4, seed=123)
    for row in lhs_samples:
        p_i = HawkesMertonParams.from_vector(row)
        pt_i = solve_point_mdp(grid=grid, params=p_i, cost_model=cm, constants=consts, tol=1e-5)
        assert np.all(imdp_res.value_function + 1e-2 >= pt_i.value_function)

    assert imdp_res.action_at(grid, h=0.15, lam=0.25) == MaintenanceAction.CONTINUE
    assert imdp_res.action_at(grid, h=2.45, lam=0.50) == MaintenanceAction.PREVENTIVE_REPLACE
    regions = imdp_res.policy_region_fractions()
    assert np.isclose(sum(regions.values()), 1.0)
    assert regions["CONTINUE"] > 0.05 and regions["PREVENTIVE_REPLACE"] > 0.05


def test_end_to_end_wp1_wp2_pipeline_hawkes_and_nhpp_fallback():
    """Verify full end-to-end WP1 -> WP2 pipeline under both self-exciting NH-Hawkes (H1)
    and non-self-exciting NHPP fallback (H0) regimes.
    """
    consts = StructuralConstants(h0=0.10, h_crit=2.50, dt=0.1667)
    grid = StateGrid2D.uniform(n_h=31, n_lam=19, lam_min=0.05, lam_max=2.50, constants=consts)
    cm = MaintenanceCostModel(discount_factor=0.99)

    p_h1 = HawkesMertonParams(a0=2.5e-3, sigma0=2.0e-3, mu0=0.25, alpha0=0.45, beta=0.90, gamma=1.35, mu_J=-2.4, sigma_J=0.60)
    sim_h1 = CoupledHawkesMertonSDE(params=p_h1, constants=consts).simulate_macro_epochs(n_paths=15, t_max=150.0, seed=55)
    ds_h1 = MacroEpochDataset.from_simulation(sim_h1, constants=consts)

    mle_h1, credal_h1, imdp_h1 = run_wp1_wp2_pipeline(ds_h1, grid=grid, cost_model=cm, constants=consts, tol=1e-4)
    assert mle_h1.hypothesis_test.active_regime == "NH-Hawkes"
    assert credal_h1.active_regime == "NH-Hawkes"
    assert imdp_h1.converged is True

    p_h0 = HawkesMertonParams(a0=2.5e-3, sigma0=2.0e-3, mu0=0.35, alpha0=0.0, beta=0.85, gamma=0.0, mu_J=-2.4, sigma_J=0.60)
    sim_h0 = CoupledHawkesMertonSDE(params=p_h0, constants=consts).simulate_macro_epochs(n_paths=12, t_max=150.0, seed=88)
    ds_h0 = MacroEpochDataset.from_simulation(sim_h0, constants=consts)

    mle_h0, credal_h0, imdp_h0 = run_wp1_wp2_pipeline(ds_h0, grid=grid, cost_model=cm, constants=consts, tol=1e-4)
    assert mle_h0.hypothesis_test.active_regime == "NHPP-Fallback"
    assert credal_h0.active_regime == "NHPP-Fallback"
    assert imdp_h0.converged is True
