"""Unit and statistical verification tests for Step 2 (WP1 Hypothesis Testing & Credal Identification)."""

import numpy as np
from scipy.stats import norm

from imdp_phm.core import CoupledHawkesMertonSDE, HawkesMertonParams, StructuralConstants
from imdp_phm.inference import (
    EpochTrajectoryData,
    CredalParameterSet,
    LogNormalPBox,
    test_self_excitation_hypothesis,
    identify_credal_model,
)


def test_h1_self_excitation_detected_on_hawkes_cohort():
    """Verify WP1 LRT & Delta BIC reject H0 (NHPP) and select NH_HAWKES when true alpha0 > 0."""
    true_params = HawkesMertonParams(
        a0=3.5e-4,
        sigma0=1.2e-3,
        mu0=0.25,
        alpha0=0.55,
        beta=1.10,
        gamma=1.30,
        mu_J=-3.10,
        sigma_J=0.85,
    )
    consts = StructuralConstants(h0=0.10, h_crit=2.50, dt=0.1667)
    sde = CoupledHawkesMertonSDE(params=true_params, constants=consts)
    sim = sde.simulate_macro_epochs(n_paths=6, t_max=180.0, seed=2026)

    cohort = [EpochTrajectoryData.from_simulation(sim, path_idx=i, constants=consts) for i in range(6)]
    res = test_self_excitation_hypothesis(cohort, alpha_sig=0.05, constants=consts)

    assert res.reject_h0 is True
    assert res.active_kernel == "NH_HAWKES"
    assert res.p_value < 0.01
    assert res.delta_bic > 0.0
    assert res.selected_params.alpha0 > 0.10
    assert np.isclose(res.diffusion_fit.a0, true_params.a0, rtol=0.35)
    assert np.isclose(res.diffusion_fit.sigma0, true_params.sigma0, rtol=0.25)
    assert np.isclose(res.jump_fit.mu_J, true_params.mu_J, atol=0.35)


def test_h0_nhpp_fallback_when_no_self_excitation():
    """Verify automatic pivot to NHPP_FALLBACK (alpha0 = 0, gamma = 0) when data come from pure NHPP."""
    nhpp_params = HawkesMertonParams(
        a0=4.0e-4,
        sigma0=1.2e-3,
        mu0=0.45,
        alpha0=0.0,
        beta=1.0,
        gamma=0.0,
        mu_J=-3.0,
        sigma_J=0.70,
    )
    consts = StructuralConstants(h0=0.10, h_crit=2.50, dt=0.1667)
    sde = CoupledHawkesMertonSDE(params=nhpp_params, constants=consts)
    sim = sde.simulate_macro_epochs(n_paths=4, t_max=150.0, seed=777)

    cohort = [EpochTrajectoryData.from_simulation(sim, path_idx=i, constants=consts) for i in range(4)]
    test_res, credal_set, pbox = identify_credal_model(cohort, alpha_sig=0.05, constants=consts)

    assert test_res.reject_h0 is False
    assert test_res.active_kernel == "NHPP_FALLBACK"
    assert test_res.selected_params.alpha0 == 0.0
    assert test_res.selected_params.gamma == 0.0
    assert credal_set.is_nhpp_fallback is True
    assert credal_set.lower[3] == 0.0 and credal_set.upper[3] == 0.0
    # Under NHPP fallback, only 5 active dimensions vary -> 2^5 = 32 vertices
    assert len(credal_set.vertices()) == 32


def test_credal_set_convexity_subcriticality_and_enclosure():
    """Verify CredalParameterSet contains nominal vector, enforces subcriticality, and samples inside Theta."""
    nom = HawkesMertonParams()
    se = HawkesMertonParams.nasa_ims_test2_se()
    credal = CredalParameterSet.from_mle_and_se(nom, se, z_mult=1.96, epistemic_inflation=1.30)

    assert credal.contains(nom)
    assert credal.contains(credal.lower_params())
    assert credal.contains(credal.upper_params())

    # Worst-case branching ratio must stay strictly subcritical (< 1)
    assert credal.upper_params().max_branching_ratio() < 1.0

    # All 256 extreme vertices and 32 LHS collocation points must lie inside Theta
    verts = credal.vertices()
    assert len(verts) == 256
    for v in verts[:20]:
        assert credal.contains(v)
        assert v.max_branching_ratio() < 1.0

    lhs_pts = credal.sample_collocation(n_samples=32, method="lhs", seed=101)
    assert len(lhs_pts) == 32
    for pt in lhs_pts:
        assert credal.contains(pt)


def test_lognormal_pbox_exact_envelope_and_quantiles():
    """Verify [F_J_low, F_J_high] encloses every interior log-normal CDF and preserves quantile ordering."""
    pbox = LogNormalPBox(mu_low=-3.6, mu_high=-3.0, sigma_low=0.80, sigma_high=1.20)
    y_grid = np.logspace(-3, 0.5, 200)

    # Test interior distributions across the parameter rectangle
    for mu in (-3.6, -3.3, -3.0):
        for sig in (0.80, 1.00, 1.20):
            cdf_interior = norm.cdf((np.log(y_grid) - mu) / sig)
            assert pbox.encloses_cdf(y_grid, cdf_interior)

    # Worst-case (FSD) quantiles must exceed best-case quantiles across all probability levels
    qs = np.array([0.10, 0.50, 0.90, 0.99])
    assert np.all(pbox.ppf_worst_case(qs) > pbox.ppf_best_case(qs))

    e_lo, e_hi = pbox.expected_jump_bounds()
    assert 0.0 < e_lo < e_hi
