"""Unit and statistical verification tests for Step 1 (WP1 Core SDE & Multiscale Pipeline)."""

import numpy as np
from scipy.stats import kstest
from imdp_phm.core import CoupledHawkesMertonSDE, HawkesMertonParams, StructuralConstants
from imdp_phm.multiscale import (
    extract_snapshot_shock_metric,
    build_macro_epoch_trajectory,
)


def test_calibrated_parameters_and_subcriticality():
    params = HawkesMertonParams()
    params.validate()
    # RESS Section 3.4 & 2.3 checks
    assert np.isclose(params.nominal_branching_ratio(), 0.35 / 0.85, atol=1e-6)
    assert params.max_branching_ratio() < 1.0  # 0.35 * 2.4 / 0.85 = 0.9882 < 1
    assert np.isclose(params.expected_jump_size(), 0.0579, atol=1e-3)


def test_infinitesimal_generator_dynkin_martingale():
    """Verify infinitesimal generator L_theta f(h, lam) against short-horizon Monte Carlo expectation."""
    params = HawkesMertonParams(a0=0.02, sigma0=0.03, mu0=0.5, alpha0=0.4, beta=1.2, gamma=1.0, mu_J=-2.5, sigma_J=0.4)
    consts = StructuralConstants(h0=0.5, h_crit=5.0, dt=1e-3)
    sde = CoupledHawkesMertonSDE(params=params, constants=consts)

    h_init, lam_init = 0.5, 0.8
    # Smooth quadratic test function f(h, lam) = h^2 + 0.5 * h * lam + 0.25 * lam^2
    f = lambda h, lam: h**2 + 0.5 * h * lam + 0.25 * (lam**2)

    l_f = sde.generator(h_init, lam_init, f)
    res = sde.simulate_macro_epochs(n_paths=250000, t_max=consts.dt, seed=12345, h_init=h_init, lam_init=lam_init)
    f_0 = f(h_init, lam_init)
    f_dt = f(res.h_paths[:, 1], res.lambda_paths[:, 1])
    empirical_rate = float(np.mean(f_dt - f_0) / consts.dt)

    assert np.isclose(empirical_rate, l_f, rtol=0.05), f"Generator {l_f} vs MC {empirical_rate}"


def test_vectorized_first_exit_statistics():
    """Verify causal ordering 0 < tau_MAJOR < tau_crit and stochastic acceleration of self-excitation (alpha0 > 0 vs alpha0 = 0)."""
    sde_hawkes = CoupledHawkesMertonSDE()
    res_h = sde_hawkes.simulate_macro_epochs(n_paths=3000, t_max=250.0, seed=0x8F5C2A4D)

    valid_lih = res_h.lead_to_imminent_hazard[~np.isnan(res_h.lead_to_imminent_hazard)]
    assert len(valid_lih) > 2500
    assert np.all(valid_lih > 0.0)
    assert float(np.mean(valid_lih >= 24.0)) > 0.90

    # Compare against NHPP baseline (alpha0 = 0): self-excitation must strictly accelerate failure
    p_nhpp = HawkesMertonParams(alpha0=0.0)
    sde_nhpp = CoupledHawkesMertonSDE(params=p_nhpp)
    res_0 = sde_nhpp.simulate_macro_epochs(n_paths=3000, t_max=250.0, seed=0x8F5C2A4D)

    mean_crit_hawkes = float(np.nanmean(res_h.tau_crit))
    mean_crit_nhpp = float(np.nanmean(res_0.tau_crit))
    assert mean_crit_hawkes < mean_crit_nhpp


def test_random_time_change_compensator_residuals():
    """Verify exact continuous-time Ogata trajectories yield Exp(1) compensator residuals (IEEE Eq. 29-30)."""
    params = HawkesMertonParams(a0=1e-4, sigma0=1e-3, mu0=1.5, alpha0=0.6, beta=1.5, gamma=1.2, mu_J=-3.5, sigma_J=0.5)
    consts = StructuralConstants(h0=0.10, h_crit=10.0)
    sde = CoupledHawkesMertonSDE(params=params, constants=consts)

    traj = sde.simulate_continuous_ogata(t_max=300.0, dt_sub=0.05, seed=99, stop_at_crit=False)
    residuals = sde.compensator_residuals(traj)
    assert len(residuals) >= 200

    ks_stat, p_val = kstest(residuals, "expon", args=(0.0, 1.0))
    assert p_val > 0.05, f"Compensator residuals rejected Exp(1): D={ks_stat:.4f}, p={p_val:.4f}"


def test_multiscale_bpfo_suppression_and_macro_aggregation():
    """Verify bandpass (2-5 kHz) + Hilbert 99.9th quantile suppresses low-frequency BPFO carrier modulation
    while detecting high-frequency structural resonance bursts on slow macro-epochs.
    """
    fs = 20000.0
    t = np.arange(20480) / fs
    rng = np.random.default_rng(7)

    # Healthy snapshot: strong shaft rotation (33.33 Hz) + kinematic BPFO (236.4 Hz) + ambient noise
    healthy_sig = (
        1.5 * np.sin(2 * np.pi * 33.33 * t)
        + 0.8 * np.sin(2 * np.pi * 236.4 * t)
        + 0.05 * rng.standard_normal(len(t))
    )
    # Damaged snapshot: same kinematic frequencies + high-frequency bearing raceway resonance burst (3.2 kHz)
    burst_env = np.exp(-0.5 * ((t - 0.5) / 0.01) ** 2)
    damaged_sig = healthy_sig + 1.2 * burst_env * np.sin(2 * np.pi * 3200.0 * t)

    v_healthy = extract_snapshot_shock_metric(healthy_sig, fs=fs)
    v_damaged = extract_snapshot_shock_metric(damaged_sig, fs=fs)
    assert v_damaged > 4.0 * v_healthy

    # Build 150 macro-epochs (first 120 nominal, last 30 degraded)
    v_series = np.array(
        [extract_snapshot_shock_metric(
            1.5 * np.sin(2 * np.pi * 33.33 * t) + 0.8 * np.sin(2 * np.pi * 236.4 * t) + 0.05 * rng.standard_normal(len(t)),
            fs=fs,
        ) for _ in range(120)]
        + [v_damaged for _ in range(30)]
    )
    feats, profile = build_macro_epoch_trajectory(v_series, nominal_snapshots=120)
    assert np.sum(feats.shock_events[:120]) == 0
    assert np.sum(feats.shock_events[120:]) == 30
    assert feats.h_hat[-1] > feats.h_hat[119]
