"""Unit and numerical verification tests for Step 3B (WP2 GP-Assisted Upper Transition Operator)."""

import numpy as np
from imdp_phm.core import HawkesMertonParams, StructuralConstants
from imdp_phm.inference import CredalParameterSet
from imdp_phm.solver import (
    MaintenanceAction,
    MaintenanceCostModel,
    StateGrid2D,
    SingleStepQuadratureOperator,
    GaussianProcessUpperOperator,
)


def _sample_credal_set() -> CredalParameterSet:
    base = HawkesMertonParams(a0=2.0e-3, sigma0=2.0e-3, mu0=0.25, alpha0=0.40, beta=0.90, gamma=1.30, mu_J=-2.4, sigma_J=0.60).to_vector()
    se = np.array([0.4e-3, 0.4e-3, 0.05, 0.08, 0.10, 0.25, 0.25, 0.12])
    return CredalParameterSet(lower=base - se, upper=base + se, max_branching_cap=0.99)


def test_gp_surrogate_upper_expectation_accuracy_and_dominance():
    """Verify GP-assisted upper operator matches direct maximization over Theta (<0.5% mean error)
    and dominates every interior parameter realization theta in Theta.
    """
    consts = StructuralConstants(h0=0.10, h_crit=2.50, dt=0.1667)
    grid = StateGrid2D.uniform(n_h=31, n_lam=21, lam_min=0.10, lam_max=2.50, constants=consts)
    cm = MaintenanceCostModel()
    op = SingleStepQuadratureOperator(grid=grid, cost_model=cm, constants=consts)
    credal = _sample_credal_set()

    h_mesh, lam_mesh = grid.meshgrid()
    v_test = 250.0 * ((h_mesh / consts.h_crit) ** 1.5) + 45.0 * lam_mesh

    gp_op = GaussianProcessUpperOperator(quad_operator=op, credal_set=credal, n_design=28, seed=42)
    res_gp = gp_op.upper_transition_surrogate(v_test, verify_exact_candidates=True)
    res_dir = gp_op.upper_transition_direct(v_test, n_random_interior=40)

    assert res_gp.gp_r2_score > 0.95, f"Low GP surrogate R2: {res_gp.gp_r2_score}"

    rel_err = np.abs(res_gp.upper_expectation - res_dir.upper_expectation) / np.maximum(np.abs(res_dir.upper_expectation), 1.0)
    assert float(np.mean(rel_err)) < 0.005 and float(np.max(rel_err)) < 0.025

    lhs_check = credal.sample_latin_hypercube(n_samples=15, seed=999)
    for row in lhs_check:
        ev_i = op.all_action_expectations(v_test, HawkesMertonParams.from_vector(row))
        assert np.all(res_gp.upper_expectation + 0.05 >= ev_i)


def test_non_vertex_interior_extremum_resolution():
    """Verify GP surrogate + L-BFGS-B resolves a true interior maximum over Theta when the value function
    peaks at an intermediate damage state (non-monotone objective over Theta where both lower and upper
    corners under-predict the interior extremum).
    """
    consts = StructuralConstants(h0=0.10, h_crit=2.50, dt=0.1667)
    grid = StateGrid2D.uniform(n_h=31, n_lam=21, lam_min=0.05, lam_max=2.50, constants=consts)
    op = SingleStepQuadratureOperator(grid=grid, constants=consts)

    base = HawkesMertonParams(a0=2.0e-3, sigma0=2.0e-3, mu0=0.35, alpha0=0.40, beta=0.90, gamma=1.30, mu_J=-2.20, sigma_J=0.45).to_vector()
    se = np.array([0.4e-3, 0.4e-3, 0.05, 0.08, 0.10, 0.25, 1.20, 0.10])
    credal = CredalParameterSet(lower=base - se, upper=base + se, max_branching_cap=0.99)

    h_mesh, lam_mesh = grid.meshgrid()
    v_nonmonotone = 500.0 * np.exp(-((h_mesh - 1.25) / 0.20) ** 2) + 20.0 * lam_mesh

    ih = 14  # h = 1.1667 (just below peak at 1.25)
    il = 8   # lam = 1.03

    gp_op = GaussianProcessUpperOperator(quad_operator=op, credal_set=credal, n_design=40, seed=17)
    res_gp = gp_op.upper_transition_surrogate(
        v_nonmonotone,
        verify_exact_candidates=True,
        extra_probe_states=[(ih, il, int(MaintenanceAction.CONTINUE))],
    )

    theta_star = res_gp.worst_case_theta[ih, il, int(MaintenanceAction.CONTINUE)]
    assert credal.contains(theta_star)

    assert credal.lower[6] + 1e-3 < theta_star[6] < credal.upper[6] - 1e-3

    verts = credal.vertices(project_subcritical=True)
    vert_vals = [
        op.expected_next_value(v_nonmonotone, HawkesMertonParams.from_vector(v), MaintenanceAction.CONTINUE)[ih, il]
        for v in verts
    ]
    max_vert = float(np.max(vert_vals))
    gp_sup = float(res_gp.upper_expectation[ih, il, int(MaintenanceAction.CONTINUE)])
    assert gp_sup > max_vert + 0.1, f"Expected strict interior gain over vertices: {gp_sup:.4f} vs {max_vert:.4f}"
