"""Verification tests for Step 2 (WP1 Hypothesis Testing H0 vs H1, MLE, Credal Set Theta, and p-box)."""

import numpy as np
from scipy.stats import norm
from imdp_phm.core import CoupledHawkesMertonSDE, HawkesMertonParams, StructuralConstants
from imdp_phm.inference import (
    MacroEpochDataset,
    HypothesisTestResult,
    MLEEstimationResult,
    fit_hawkes_merton_mle,
    CredalParameterSet,
    JumpSeverityPBox,
)


def test_decoupled_mle_and_hawkes_detection():
    """Verify MLE recovers continuous wear, jump severity, and rejects H0 on self-exciting trajectories."""
    consts = StructuralConstants(h0=0.10, h_crit=3.0, dt=0.1667)
    p_hawkes = HawkesMertonParams(
        a0=4.5e-4, sigma0=1.2e-3, mu0=0.25, alpha0=0.55, beta=0.90, gamma=1.10, mu_J=-2.8, sigma_J=0.55
    )
    sim = CoupledHawkesMertonSDE(params=p_hawkes, constants=consts).simulate_macro_epochs(
        n_paths=15, t_max=150.0, seed=101
    )
    ds = MacroEpochDataset.from_simulation(sim, constants=consts)
    mle = fit_hawkes_merton_mle(ds, constants=consts)

    assert np.isclose(mle.effective_params.a0, p_hawkes.a0, rtol=0.20)
    assert np.isclose(mle.effective_params.mu_J, p_hawkes.mu_J, atol=0.20)
    assert mle.hypothesis_test.reject_h0 is True
    assert mle.hypothesis_test.active_regime == "NH-Hawkes"
    assert mle.hypothesis_test.p_value < 0.01
    assert mle.hypothesis_test.delta_bic > 2.0


def test_nhpp_fallback_when_no_self_excitation():
    """Verify WP1 hypothesis test falls back cleanly to NHPP (alpha0=0, gamma=0) when alpha0=0."""
    consts = StructuralConstants(h0=0.10, h_crit=3.0, dt=0.1667)
    p_nhpp = HawkesMertonParams(
        a0=3.5e-4, sigma0=1.1e-3, mu0=0.40, alpha0=0.0, beta=0.90, gamma=0.0, mu_J=-3.0, sigma_J=0.60
    )
    sim = CoupledHawkesMertonSDE(params=p_nhpp, constants=consts).simulate_macro_epochs(
        n_paths=12, t_max=140.0, seed=303
    )
    ds = MacroEpochDataset.from_simulation(sim, constants=consts)
    mle = fit_hawkes_merton_mle(ds, constants=consts)

    assert mle.hypothesis_test.reject_h0 is False
    assert mle.hypothesis_test.active_regime == "NHPP-Fallback"
    assert mle.effective_params.alpha0 == 0.0
    assert mle.effective_params.gamma == 0.0


def test_credal_parameter_set_and_pbox_bounds():
    """Verify CredalParameterSet enforces subcriticality and JumpSeverityPBox bounds all interior CDFs."""
    consts = StructuralConstants(h0=0.10, h_crit=2.50, dt=0.1667)
    true_p = HawkesMertonParams(a0=2.8e-3, sigma0=2.0e-3, mu0=0.30, alpha0=0.40, beta=0.90, gamma=1.30, mu_J=-2.3, sigma_J=0.58)
    sim = CoupledHawkesMertonSDE(params=true_p, constants=consts).simulate_macro_epochs(
        n_paths=6, t_max=150.0, seed=404
    )
    ds = MacroEpochDataset.from_simulation(sim, constants=consts)
    mle = fit_hawkes_merton_mle(ds, constants=consts)
    credal = CredalParameterSet.from_mle_result(mle, n_trajectories=6, z_mult=1.96)

    assert np.all(credal.lower <= credal.upper)
    lhs = credal.sample_latin_hypercube(n_samples=24, seed=77)
    assert lhs.shape == (24, 8)
    for row in lhs:
        assert row[3] * (1.0 + row[5]) / row[4] <= credal.max_branching_cap + 1e-9

    pb = credal.jump_pbox()
    assert isinstance(pb, JumpSeverityPBox)
    y_grid = np.linspace(0.01, 0.5, 50)
    ln_y = np.log(y_grid)
    cdf_low = np.minimum(norm.cdf((ln_y - pb.mu_high) / pb.sigma_low), norm.cdf((ln_y - pb.mu_high) / pb.sigma_high))
    cdf_high = np.maximum(norm.cdf((ln_y - pb.mu_low) / pb.sigma_low), norm.cdf((ln_y - pb.mu_low) / pb.sigma_high))
    assert np.all(cdf_low <= cdf_high + 1e-12)
