"""Unit and behavioral verification tests for Step 4A (WP3 Baseline Policies: IMDP, MLE-MDP, Threshold, LSTM-RUL)."""

import numpy as np
from imdp_phm.core import CoupledHawkesMertonSDE, HawkesMertonParams, StructuralConstants
from imdp_phm.inference import MacroEpochDataset
from imdp_phm.solver import MaintenanceAction, MaintenanceCostModel, StateGrid2D
from imdp_phm.benchmark import (
    IMDPPolicy,
    MLEMDPPolicy,
    ThresholdPolicy,
    LSTMRULPolicy,
)


def test_threshold_policy_ordering_and_validation():
    """Verify ThresholdPolicy respects damage & hazard alert boundaries."""
    tp = ThresholdPolicy(h_inspect=1.35, h_replace=2.00, lam_inspect=0.80, h_crit=2.50)
    assert tp.step(h=0.25, lam=0.20) == MaintenanceAction.CONTINUE
    assert tp.step(h=0.40, lam=0.95) == MaintenanceAction.NDE_INSPECT
    assert tp.step(h=1.50, lam=0.25) == MaintenanceAction.NDE_INSPECT
    assert tp.step(h=2.05, lam=0.30) == MaintenanceAction.PREVENTIVE_REPLACE


def test_lstm_rul_policy_training_and_degradation_escalation():
    """Verify LSTMRULPolicy trains on run-to-failure cohorts, decreases loss, predicts shrinking RUL
    along a degrading trajectory, and escalates from CONTINUE -> NDE_INSPECT -> PREVENTIVE_REPLACE.
    """
    consts = StructuralConstants(h0=0.10, h_crit=2.50, dt=0.1667)
    params = HawkesMertonParams(a0=3.5e-3, sigma0=2.5e-3, mu0=0.35, alpha0=0.45, beta=0.90, gamma=1.35, mu_J=-2.1, sigma_J=0.55)
    sim = CoupledHawkesMertonSDE(params=params, constants=consts).simulate_macro_epochs(n_paths=24, t_max=140.0, seed=101)
    ds = MacroEpochDataset.from_simulation(sim, constants=consts)

    lstm_pol = LSTMRULPolicy(
        constants=consts,
        seq_len=12,
        hidden_dim=24,
        max_rul_hours=80.0,
        rul_inspect_hours=14.0,
        rul_replace_hours=6.0,
        seed=42,
    )
    rep = lstm_pol.fit_from_dataset(ds, epochs=30, lr=4e-3, stride=2)
    assert rep.n_sequences > 50
    assert rep.final_train_loss < rep.loss_history[0]
    assert rep.train_rmse_hours < 20.0

    fail_paths = np.flatnonzero(~np.isnan(sim.tau_crit))
    assert len(fail_paths) > 0
    idx = int(fail_paths[0])
    k_fail = int(np.flatnonzero(sim.h_paths[idx, 1:] >= consts.h_crit)[0]) + 1

    lstm_pol.reset()
    ruls = []
    actions = []
    for k in range(k_fail):
        h_k = float(sim.h_paths[idx, k + 1])
        lam_k = float(sim.lambda_paths[idx, k + 1])
        dn_k = int(sim.jump_counts[idx, k])
        act = lstm_pol.step(h=h_k, lam=lam_k, jump_count=dn_k)
        ruls.append(lstm_pol.predict_rul_hours())
        actions.append(act)

    assert np.mean(ruls[:15]) > np.mean(ruls[-10:]) + 5.0
    assert actions[0] == MaintenanceAction.CONTINUE
    assert actions[-1] == MaintenanceAction.PREVENTIVE_REPLACE

    lstm_pol.reset()
    assert lstm_pol.step(h=0.12, lam=0.20, jump_count=0) == MaintenanceAction.CONTINUE


def test_imdp_vs_mlemdep_epistemic_conservatism():
    """Verify IMDPPolicy upper value function dominates MLEMDPPolicy everywhere on the 2D state space
    and produces valid protective policies across pristine and critical regimes.
    """
    consts = StructuralConstants(h0=0.10, h_crit=2.50, dt=0.1667)
    grid = StateGrid2D.uniform(n_h=35, n_lam=21, lam_min=0.05, lam_max=2.50, constants=consts)
    cm = MaintenanceCostModel(c_inspection=20.0, c_preventive=230.0, c_failure=3500.0, discount_factor=0.99)

    params = HawkesMertonParams(a0=2.5e-3, sigma0=2.0e-3, mu0=0.25, alpha0=0.45, beta=0.90, gamma=1.35, mu_J=-2.3, sigma_J=0.60)
    sim = CoupledHawkesMertonSDE(params=params, constants=consts).simulate_macro_epochs(n_paths=12, t_max=150.0, seed=77)
    ds = MacroEpochDataset.from_simulation(sim, constants=consts)

    imdp_pol, _, credal, imdp_res = IMDPPolicy.from_dataset(ds, grid=grid, cost_model=cm, constants=consts, tol=1e-4, seed=42)
    mle_pol, _, point_res = MLEMDPPolicy.from_dataset(ds, grid=grid, cost_model=cm, constants=consts, tol=1e-5)

    assert credal.active_regime == "NH-Hawkes"
    assert imdp_pol.step(h=0.15, lam=0.25) == MaintenanceAction.CONTINUE
    assert mle_pol.step(h=0.15, lam=0.25) == MaintenanceAction.CONTINUE
    assert imdp_pol.step(h=2.45, lam=0.50) == MaintenanceAction.PREVENTIVE_REPLACE
    assert mle_pol.step(h=2.45, lam=0.50) == MaintenanceAction.PREVENTIVE_REPLACE

    assert np.all(imdp_res.value_function + 1e-3 >= point_res.value_function)
    assert imdp_res.value_at(grid, 0.10, 0.25) > point_res.value_at(grid, 0.10, 0.25)
