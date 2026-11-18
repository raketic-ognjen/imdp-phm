"""Verification tests for Step 4B (WP3 Closed-Loop Evaluator & Competitive Benchmark)."""

import numpy as np
from imdp_phm.core import CoupledHawkesMertonSDE, HawkesMertonParams, StructuralConstants
from imdp_phm.inference import MacroEpochDataset
from imdp_phm.solver import MaintenanceCostModel, StateGrid2D
from imdp_phm.benchmark import (
    ThresholdPolicy,
    compute_cvar,
    evaluate_policy_closed_loop,
    run_wp3_competitive_benchmark,
)


def test_cvar_and_closed_loop_metrics_consistency():
    """Verify CVaR_95 >= mean_cost and failure + replacement + censored rates stay bounded in [0, 1]."""
    consts = StructuralConstants(h0=0.10, h_crit=2.50, dt=0.1667)
    cm = MaintenanceCostModel(c_inspection=20.0, c_preventive=240.0, c_failure=3500.0, discount_factor=0.995)
    params = HawkesMertonParams(a0=3.0e-3, sigma0=2.2e-3, mu0=0.35, alpha0=0.42, beta=0.90, gamma=1.35, mu_J=-2.2, sigma_J=0.60)
    tp = ThresholdPolicy(h_inspect=1.35, h_replace=2.00, lam_inspect=0.85, h_crit=2.50)

    res = evaluate_policy_closed_loop(tp, params, cost_model=cm, constants=consts, n_episodes=80, horizon_hours=140.0, seed=11)
    assert res.cvar_95_cost >= res.mean_total_cost
    assert 0.0 <= res.failure_rate <= 1.0
    assert 0.0 <= res.premature_replacement_rate <= res.mean_replacements + 1e-12
    assert res.mean_service_life_hours > 0.0


def test_wp3_four_way_benchmark_under_load_shift():
    """Run full 4-way WP3 benchmark (IMDP-PHM vs MLE-MDP vs LSTM-RUL vs Threshold) under scarce
    training data and Load-Shift stress testing; verify IMDP-PHM suppresses uncontained failures
    and tail CVaR_95 cost under regime shift.
    """
    consts = StructuralConstants(h0=0.10, h_crit=2.50, dt=0.1667)
    grid = StateGrid2D.uniform(n_h=33, n_lam=19, lam_min=0.05, lam_max=2.50, constants=consts)
    cm = MaintenanceCostModel(c_inspection=20.0, c_preventive=240.0, c_failure=3500.0, discount_factor=0.995)

    train_params = HawkesMertonParams(a0=2.8e-3, sigma0=2.0e-3, mu0=0.30, alpha0=0.40, beta=0.90, gamma=1.30, mu_J=-2.3, sigma_J=0.58)
    sim_train = CoupledHawkesMertonSDE(params=train_params, constants=consts).simulate_macro_epochs(n_paths=12, t_max=150.0, seed=55)
    ds_train = MacroEpochDataset.from_simulation(sim_train, constants=consts)

    suites = run_wp3_competitive_benchmark(
        train_dataset=ds_train,
        nominal_test_params=train_params,
        grid=grid,
        cost_model=cm,
        constants=consts,
        n_eval_episodes=100,
        horizon_hours=140.0,
        lstm_epochs=20,
        seed=42,
    )

    assert set(suites.keys()) == {"Nominal", "Load-Shift"}
    for reg_name, suite in suites.items():
        assert set(suite.metrics_by_policy.keys()) == {"IMDP-PHM", "MLE-MDP", "LSTM-RUL", "Threshold"}
        imdp_m = suite.metrics_by_policy["IMDP-PHM"]
        thresh_m = suite.metrics_by_policy["Threshold"]
        assert imdp_m.failure_rate <= thresh_m.failure_rate + 0.05
        assert imdp_m.cvar_95_cost <= thresh_m.cvar_95_cost

    ls_imdp = suites["Load-Shift"].metrics_by_policy["IMDP-PHM"]
    ls_thresh = suites["Load-Shift"].metrics_by_policy["Threshold"]
    ls_lstm = suites["Load-Shift"].metrics_by_policy["LSTM-RUL"]
    assert ls_imdp.failure_rate <= ls_thresh.failure_rate
    assert ls_imdp.failure_rate < ls_lstm.failure_rate
    assert ls_imdp.cvar_95_cost < ls_thresh.cvar_95_cost
    assert ls_imdp.cvar_95_cost < ls_lstm.cvar_95_cost
