"""WP3 Closed-Loop Monte Carlo Policy Evaluator & Load-Shift Stress Benchmark (Concept Note WP3).

Evaluates competing maintenance policies (IMDP-PHM, MLE-MDP, LSTM-RUL, Threshold) in closed-loop
stochastic rollouts with action feedback (NDE hazard relief, preventive renewal, absorbing failure)
and computes:
- Mean discounted & undiscounted lifecycle cost
- Tail cost risk CVaR_{95%}
- Uncontained failure rate P(H_t >= H_crit)
- Premature replacement rate (replacement when H_t < 0.50 * H_crit)
- Mean inspection count & Mean Time Between Replacements (MTBR)
- Decision Lead-to-Imminent-Hazard margin Delta_tau_LIH [h]
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from imdp_phm.benchmark.baselines import (
    IMDPPolicy,
    LSTMRULPolicy,
    MLEMDPPolicy,
    MaintenancePolicy,
    ThresholdPolicy,
)
from imdp_phm.core.params import HawkesMertonParams, StructuralConstants
from imdp_phm.core.sde import CoupledHawkesMertonSDE
from imdp_phm.inference.mle import MacroEpochDataset
from imdp_phm.solver.grid import MaintenanceAction, MaintenanceCostModel, StateGrid2D


@dataclass(frozen=True)
class PolicyEvaluationMetrics:
    """Comprehensive WP3 operational, economic, and tail-risk metrics for one policy."""
    policy_name: str
    regime_name: str
    n_episodes: int
    mean_total_cost: float
    std_total_cost: float
    cvar_95_cost: float
    mean_discounted_cost: float
    failure_rate: float
    premature_replacement_rate: float
    mean_inspections: float
    mean_replacements: float
    mean_service_life_hours: float
    mean_lead_time_hours: float
    episode_costs: np.ndarray = field(repr=False)


@dataclass
class WP3BenchmarkSuiteResult:
    """Aggregated comparison across all policies and stress regimes."""
    regime_name: str
    true_test_params: HawkesMertonParams
    metrics_by_policy: dict[str, PolicyEvaluationMetrics]

    def summary_table(self) -> list[dict[str, float | str]]:
        rows = []
        for name, m in self.metrics_by_policy.items():
            rows.append({
                "policy": name,
                "regime": m.regime_name,
                "mean_cost": round(m.mean_total_cost, 2),
                "cvar_95": round(m.cvar_95_cost, 2),
                "discounted_cost": round(m.mean_discounted_cost, 2),
                "failure_rate": round(m.failure_rate, 4),
                "premature_rep_rate": round(m.premature_replacement_rate, 4),
                "mean_inspections": round(m.mean_inspections, 2),
                "lead_time_h": round(m.mean_lead_time_hours, 2),
            })
        return rows


def compute_cvar(costs: np.ndarray, alpha: float = 0.95) -> float:
    """Compute Conditional Value-at-Risk CVaR_alpha (expected loss in worst (1 - alpha) tail)."""
    arr = np.sort(np.asarray(costs, dtype=np.float64))
    if len(arr) == 0:
        return 0.0
    var_idx = int(np.floor(alpha * len(arr)))
    var_idx = min(max(var_idx, 0), len(arr) - 1)
    return float(np.mean(arr[var_idx:]))


def evaluate_policy_closed_loop(
    policy: MaintenancePolicy,
    true_params: HawkesMertonParams,
    cost_model: Optional[MaintenanceCostModel] = None,
    constants: Optional[StructuralConstants] = None,
    n_episodes: int = 200,
    horizon_hours: float = 160.0,
    regime_name: str = "Nominal",
    seed: int = 2026,
) -> PolicyEvaluationMetrics:
    """Run closed-loop stochastic rollouts of `policy` on coupled Hawkes-Merton degradation
    governed by `true_params`. Uses identical Common Random Numbers (CRN) across policies
    for variance-reduced paired benchmarking.
    """
    cm = cost_model or MaintenanceCostModel()
    consts = constants or StructuralConstants()
    sde = CoupledHawkesMertonSDE(params=true_params, constants=consts)

    dt = consts.dt
    h_crit = consts.h_crit
    k_max = int(np.floor(horizon_hours / dt))
    rho = float(np.exp(-true_params.beta * dt))
    sqrt_dt = np.sqrt(dt)

    rng = np.random.Generator(np.random.PCG64(seed))
    dw_all = rng.standard_normal((n_episodes, k_max))
    u_jump_all = rng.random((n_episodes, k_max))
    z_jump_all = rng.standard_normal((n_episodes, k_max))

    ep_total_costs = np.zeros(n_episodes, dtype=np.float64)
    ep_disc_costs = np.zeros(n_episodes, dtype=np.float64)
    ep_failures = np.zeros(n_episodes, dtype=np.float64)
    ep_premature_reps = np.zeros(n_episodes, dtype=np.float64)
    ep_inspections = np.zeros(n_episodes, dtype=np.float64)
    ep_replacements = np.zeros(n_episodes, dtype=np.float64)
    ep_service_hours = np.zeros(n_episodes, dtype=np.float64)
    ep_lead_times = []

    for ep in range(n_episodes):
        policy.reset()
        h = consts.h0
        lam = true_params.mu0
        last_jump_count = 0
        disc = 1.0
        first_alert_t: Optional[float] = None
        terminal_t: float = horizon_hours

        for k in range(k_max):
            t_curr = k * dt
            action = policy.step(h=h, lam=lam, jump_count=last_jump_count)

            if action != MaintenanceAction.CONTINUE and first_alert_t is None:
                first_alert_t = t_curr

            c_k = cm.step_cost(h=h, action=action, h_crit=h_crit)
            ep_total_costs[ep] += c_k
            ep_disc_costs[ep] += disc * c_k
            disc *= cm.discount_factor

            if h >= h_crit - 1e-12:
                ep_failures[ep] = 1.0
                terminal_t = t_curr
                break

            if action == MaintenanceAction.PREVENTIVE_REPLACE:
                ep_replacements[ep] = 1.0
                if h < 0.50 * h_crit:
                    ep_premature_reps[ep] = 1.0
                terminal_t = t_curr
                break

            if action == MaintenanceAction.NDE_INSPECT:
                ep_inspections[ep] += 1.0
                lam_eff = true_params.mu0 + (1.0 - cm.inspection_hazard_relief) * max(0.0, lam - true_params.mu0)
            else:
                lam_eff = lam

            p_jump = 1.0 - np.exp(-lam_eff * dt)
            jumped = bool(u_jump_all[ep, k] < p_jump)
            y_j = float(np.exp(true_params.mu_J + true_params.sigma_J * z_jump_all[ep, k])) if jumped else 0.0
            a_step = float(sde.drift(h)) * dt
            sig_step = float(sde.volatility(h)) * sqrt_dt
            alpha_h = float(sde.excitation_alpha(h))

            h_next = max(0.0, h + a_step + sig_step * float(dw_all[ep, k]) + y_j)
            lam_next = true_params.mu0 + (lam_eff - true_params.mu0) * rho + (alpha_h if jumped else 0.0)

            h = h_next
            lam = lam_next
            last_jump_count = 1 if jumped else 0

            if h >= h_crit:
                c_fail = cm.step_cost(h=h, action=MaintenanceAction.CONTINUE, h_crit=h_crit)
                ep_total_costs[ep] += c_fail
                ep_disc_costs[ep] += disc * c_fail
                ep_failures[ep] = 1.0
                terminal_t = (k + 1) * dt
                break

        ep_service_hours[ep] = terminal_t
        if first_alert_t is not None:
            ep_lead_times.append(max(0.0, terminal_t - first_alert_t))

    mean_lead = float(np.mean(ep_lead_times)) if len(ep_lead_times) > 0 else 0.0

    return PolicyEvaluationMetrics(
        policy_name=policy.name,
        regime_name=regime_name,
        n_episodes=n_episodes,
        mean_total_cost=float(np.mean(ep_total_costs)),
        std_total_cost=float(np.std(ep_total_costs)),
        cvar_95_cost=compute_cvar(ep_total_costs, alpha=0.95),
        mean_discounted_cost=float(np.mean(ep_disc_costs)),
        failure_rate=float(np.mean(ep_failures)),
        premature_replacement_rate=float(np.mean(ep_premature_reps)),
        mean_inspections=float(np.mean(ep_inspections)),
        mean_replacements=float(np.mean(ep_replacements)),
        mean_service_life_hours=float(np.mean(ep_service_hours)),
        mean_lead_time_hours=mean_lead,
        episode_costs=ep_total_costs,
    )


def run_wp3_competitive_benchmark(
    train_dataset: MacroEpochDataset,
    nominal_test_params: HawkesMertonParams,
    load_shift_test_params: Optional[HawkesMertonParams] = None,
    grid: Optional[StateGrid2D] = None,
    cost_model: Optional[MaintenanceCostModel] = None,
    constants: Optional[StructuralConstants] = None,
    n_eval_episodes: int = 160,
    horizon_hours: float = 150.0,
    lstm_epochs: int = 25,
    seed: int = 42,
) -> dict[str, WP3BenchmarkSuiteResult]:
    """Train/calibrate all 4 policies on `train_dataset` and benchmark them across:
    1. 'Nominal': out-of-sample trajectories under nominal parameters.
    2. 'Load-Shift': severe operational load shift / regime escalation (Concept Note WP3).
    """
    consts = constants or StructuralConstants()
    g = grid or StateGrid2D.uniform(n_h=35, n_lam=21, lam_min=0.05, lam_max=2.50, constants=consts)
    cm = cost_model or MaintenanceCostModel()

    imdp_pol, _, _, _ = IMDPPolicy.from_dataset(train_dataset, grid=g, cost_model=cm, constants=consts, tol=1e-4, seed=seed)
    mle_pol, _, _ = MLEMDPPolicy.from_dataset(train_dataset, grid=g, cost_model=cm, constants=consts, tol=1e-5)
    thresh_pol = ThresholdPolicy(h_inspect=0.55 * consts.h_crit, h_replace=0.80 * consts.h_crit, lam_inspect=0.85, h_crit=consts.h_crit)
    lstm_pol = LSTMRULPolicy(constants=consts, seq_len=12, hidden_dim=24, max_rul_hours=90.0, rul_inspect_hours=18.0, rul_replace_hours=8.0, seed=seed)
    lstm_pol.fit_from_dataset(train_dataset, epochs=lstm_epochs, lr=4e-3, stride=2)

    policies: list[MaintenancePolicy] = [imdp_pol, mle_pol, lstm_pol, thresh_pol]

    if load_shift_test_params is None:
        p = nominal_test_params
        load_shift_test_params = HawkesMertonParams(
            a0=p.a0 * 1.45,
            sigma0=p.sigma0 * 1.30,
            mu0=p.mu0 * 1.35,
            alpha0=min(p.alpha0 * 1.40, 0.85 * p.beta / (1.0 + p.gamma)),
            beta=p.beta * 0.90,
            gamma=p.gamma * 1.20,
            mu_J=p.mu_J + 0.30,
            sigma_J=p.sigma_J * 1.15,
        )

    results: dict[str, WP3BenchmarkSuiteResult] = {}
    for reg_name, test_p, reg_seed in [
        ("Nominal", nominal_test_params, seed + 1000),
        ("Load-Shift", load_shift_test_params, seed + 2000),
    ]:
        metrics_map: dict[str, PolicyEvaluationMetrics] = {}
        for pol in policies:
            metrics_map[pol.name] = evaluate_policy_closed_loop(
                policy=pol,
                true_params=test_p,
                cost_model=cm,
                constants=consts,
                n_episodes=n_eval_episodes,
                horizon_hours=horizon_hours,
                regime_name=reg_name,
                seed=reg_seed,
            )
        results[reg_name] = WP3BenchmarkSuiteResult(
            regime_name=reg_name,
            true_test_params=test_p,
            metrics_by_policy=metrics_map,
        )
    return results
