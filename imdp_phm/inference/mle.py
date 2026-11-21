"""Decoupled & Joint MLE Parameter Estimation and H0 vs H1 Hypothesis Testing (Concept Note WP1 & RESS Sec. 3.4)."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from imdp_phm.core.params import HawkesMertonParams, StructuralConstants
from imdp_phm.core.sde import DiscreteSimulationResult
from imdp_phm.multiscale.features import SnapshotFeatures


@dataclass
class MacroEpochDataset:
    """Epoch-aggregated run-to-failure dataset across M >= 1 trajectories."""
    dt: float
    h_paths: np.ndarray          # (M, K + 1) cumulative damage trajectories
    jump_indicators: np.ndarray  # (M, K) binary jump events dN_{m,k} in {0, 1}
    jump_sizes: np.ndarray       # 1D array of observed positive jump volumes Y_j > 0
    active_mask: np.ndarray      # (M, K) bool mask of active pre-failure epochs

    @classmethod
    def from_simulation(
        cls,
        sim: DiscreteSimulationResult,
        constants: Optional[StructuralConstants] = None,
        max_paths: Optional[int] = None,
    ) -> "MacroEpochDataset":
        consts = constants or StructuralConstants()
        h = sim.h_paths if max_paths is None else sim.h_paths[:max_paths]
        jumps = (sim.jump_counts if max_paths is None else sim.jump_counts[:max_paths]) > 0
        m_paths, k_steps = jumps.shape
        dt = float(sim.time_grid[1] - sim.time_grid[0]) if len(sim.time_grid) > 1 else consts.dt

        active = h[:, :-1] < consts.h_crit
        dh = h[:, 1:] - h[:, :-1]
        y_obs = dh[jumps & active]
        y_obs = y_obs[y_obs > 1e-9]

        return cls(
            dt=dt,
            h_paths=np.asarray(h, dtype=np.float64),
            jump_indicators=jumps.astype(np.int32),
            jump_sizes=np.asarray(y_obs, dtype=np.float64),
            active_mask=active.astype(bool),
        )

    @classmethod
    def from_features(
        cls,
        features: SnapshotFeatures,
        constants: Optional[StructuralConstants] = None,
    ) -> "MacroEpochDataset":
        consts = constants or StructuralConstants()
        dt = float(features.epoch_times[1] - features.epoch_times[0]) if len(features.epoch_times) > 1 else consts.dt
        h_full = np.concatenate([[consts.h0], features.h_hat])[None, :]
        jumps = (features.shock_events > 0)[None, :]
        active = h_full[:, :-1] < consts.h_crit
        dh = h_full[:, 1:] - h_full[:, :-1]
        y_obs = dh[jumps & active]
        y_obs = y_obs[y_obs > 1e-9]
        return cls(
            dt=dt,
            h_paths=h_full,
            jump_indicators=jumps.astype(np.int32),
            jump_sizes=y_obs,
            active_mask=active,
        )


@dataclass(frozen=True)
class HypothesisTestResult:
    """Result of WP1 statistical test H0: alpha0 = 0 (NHPP baseline) vs H1: alpha0 > 0 (NH-Hawkes)."""
    loglik_h0: float
    loglik_h1: float
    loglik_gain: float
    lrt_statistic: float
    p_value: float
    delta_bic: float              # BIC(H0) - BIC(H1); positive => evidence favors H1
    n_active_epochs: int
    reject_h0: bool
    active_regime: str            # "NH-Hawkes" or "NHPP-Fallback"
    nhpp_mu0: float
    nhpp_gamma_trend: float


@dataclass
class MLEEstimationResult:
    """Complete 8-parameter MLE result with Fisher standard errors and hypothesis test outcome."""
    params_h1: HawkesMertonParams
    effective_params: HawkesMertonParams
    standard_errors: np.ndarray
    covariance_matrix: np.ndarray
    hypothesis_test: HypothesisTestResult
    total_loglik: float


def fit_continuous_wear(
    dataset: MacroEpochDataset,
    constants: Optional[StructuralConstants] = None,
) -> tuple[float, float]:
    """Exact weighted Gaussian MLE for baseline wear drift a0 and volatility sigma0
    over non-jump active epochs (RESS Eq. 18).
    """
    consts = constants or StructuralConstants()
    dt = dataset.dt
    h_prev = dataset.h_paths[:, :-1]
    dh = dataset.h_paths[:, 1:] - h_prev
    non_jump_active = dataset.active_mask & (dataset.jump_indicators == 0)

    if int(np.sum(non_jump_active)) < 5:
        defaults = HawkesMertonParams()
        return defaults.a0, defaults.sigma0

    h_sub = np.clip(h_prev[non_jump_active], 0.0, consts.h_crit) / consts.h_crit
    dh_sub = dh[non_jump_active]

    w_a = 1.0 + consts.zeta_a * (h_sub ** consts.theta_a)
    w_sig = 1.0 + consts.zeta_sigma * h_sub

    denom = dt * float(np.sum((w_a ** 2) / w_sig))
    a0_hat = float(np.sum((w_a / w_sig) * dh_sub)) / max(denom, 1e-18)
    a0_hat = max(a0_hat, 1e-7)

    resid_sq = ((dh_sub - a0_hat * w_a * dt) ** 2) / w_sig
    sigma0_sq = float(np.mean(resid_sq)) / dt
    sigma0_hat = float(np.sqrt(max(sigma0_sq, 1e-14)))
    return a0_hat, sigma0_hat


def fit_lognormal_jumps(jump_sizes: np.ndarray) -> tuple[float, float]:
    """Exact MLE for log-normal jump severity ln(Y) ~ N(mu_J, sigma_J^2) (RESS Eq. 21)."""
    y = np.asarray(jump_sizes, dtype=np.float64)
    y = y[y > 0.0]
    if len(y) == 0:
        defaults = HawkesMertonParams()
        return defaults.mu_J, defaults.sigma_J
    log_y = np.log(y)
    mu_j = float(np.mean(log_y))
    if len(y) < 2:
        sigma_j = HawkesMertonParams().sigma_J
    else:
        sigma_j = float(np.sqrt(np.mean((log_y - mu_j) ** 2)))
        sigma_j = max(sigma_j, 1e-3)
    return mu_j, sigma_j


def _nhpp_epoch_loglik(
    theta_h0: np.ndarray,
    h_prev: np.ndarray,
    jumps: np.ndarray,
    active: np.ndarray,
    dt: float,
    h_crit: float,
) -> float:
    """Log-likelihood of NHPP trend baseline (alpha0 = 0): lambda(t) = mu0 * (1 + gamma_nhpp * (H/H_crit)^2)."""
    mu0, gamma_nhpp = float(theta_h0[0]), float(theta_h0[1])
    if mu0 <= 1e-9 or gamma_nhpp < 0.0:
        return -1e18

    h_norm_sq = (np.clip(h_prev, 0.0, h_crit) / h_crit) ** 2
    lam = mu0 * (1.0 + gamma_nhpp * h_norm_sq)
    p = np.clip(1.0 - np.exp(-lam * dt), 1e-15, 1.0 - 1e-15)
    ll = np.where(active, jumps * np.log(p) + (1.0 - jumps) * np.log(1.0 - p), 0.0)
    return float(np.sum(ll))


def _hawkes_epoch_loglik(
    theta_h1: np.ndarray,
    h_prev: np.ndarray,
    jumps: np.ndarray,
    active: np.ndarray,
    dt: float,
    h_crit: float,
) -> float:
    """Exact recursive O(1)-per-step log-likelihood of state-dependent NH-Hawkes process over macro-epochs."""
    mu0, alpha0, beta, gamma = (
        float(theta_h1[0]),
        float(theta_h1[1]),
        float(theta_h1[2]),
        float(theta_h1[3]),
    )
    if mu0 <= 1e-9 or alpha0 < 0.0 or beta <= 1e-6 or gamma < 0.0:
        return -1e18
    branch_max = alpha0 * (1.0 + gamma) / beta
    if branch_max >= 0.995:
        return -1e9 * ((branch_max - 0.995) ** 2) - 1e6

    m_paths, k_steps = jumps.shape
    rho = float(np.exp(-beta * dt))
    h_norm_sq = (np.clip(h_prev, 0.0, h_crit) / h_crit) ** 2
    alpha_mat = alpha0 * (1.0 + gamma * h_norm_sq)

    lam = np.full(m_paths, mu0, dtype=np.float64)
    total_ll = 0.0

    for k in range(k_steps):
        act_k = active[:, k]
        if not np.any(act_k):
            break
        p_k = np.clip(1.0 - np.exp(-lam[act_k] * dt), 1e-15, 1.0 - 1e-15)
        dn_k = jumps[act_k, k]
        total_ll += float(np.sum(dn_k * np.log(p_k) + (1.0 - dn_k) * np.log(1.0 - p_k)))
        lam = mu0 + (lam - mu0) * rho + jumps[:, k] * alpha_mat[:, k]

    return total_ll


def fit_point_process_and_test(
    dataset: MacroEpochDataset,
    constants: Optional[StructuralConstants] = None,
    significance_level: float = 0.01,
    min_delta_bic: float = 2.0,
) -> tuple[tuple[float, float, float, float], HypothesisTestResult]:
    """Fit H0 (NHPP trend baseline) and H1 (NH-Hawkes) on epoch-aggregated trajectories,
    perform boundary-corrected LRT and Delta-BIC model selection, and trigger automatic NHPP fallback
    if H0 is not rejected (Concept Note Revision 3, Section 1 & Section 5).
    """
    consts = constants or StructuralConstants()
    h_prev = dataset.h_paths[:, :-1]
    jumps = dataset.jump_indicators.astype(np.float64)
    active = dataset.active_mask
    dt = dataset.dt
    n_obs = int(np.sum(active))

    if n_obs < 10:
        defaults = HawkesMertonParams()
        ht = HypothesisTestResult(
            loglik_h0=0.0,
            loglik_h1=0.0,
            loglik_gain=0.0,
            lrt_statistic=0.0,
            p_value=1.0,
            delta_bic=0.0,
            n_active_epochs=n_obs,
            reject_h0=False,
            active_regime="NHPP-Fallback",
            nhpp_mu0=defaults.mu0,
            nhpp_gamma_trend=0.0,
        )
        return (defaults.mu0, 0.0, defaults.beta, 0.0), ht

    emp_rate = max(float(np.sum(jumps[active])) / (n_obs * dt), 1e-3)
    rate_cap = max(20.0, 5.0 * emp_rate, 1.5 / max(dt, 1e-5))
    beta_cap = max(20.0, 2.0 / max(dt, 1e-5))
    beta_base = min(max(0.85, 0.15 / max(dt, 1e-5)), 0.5 * beta_cap)

    best_h0_res = None
    for g0 in (0.0, 0.5, 2.0, 6.0):
        res0 = minimize(
            lambda th: -_nhpp_epoch_loglik(th, h_prev, jumps, active, dt, consts.h_crit),
            x0=np.array([min(emp_rate / (1.0 + 0.3 * g0), 0.95 * rate_cap), g0], dtype=np.float64),
            method="L-BFGS-B",
            bounds=[(1e-5, rate_cap), (0.0, 50.0)],
        )
        if best_h0_res is None or res0.fun < best_h0_res.fun:
            best_h0_res = res0

    mu0_h0, gamma_h0 = float(best_h0_res.x[0]), float(best_h0_res.x[1])
    ll_h0 = -float(best_h0_res.fun)

    best_h1_res = None
    starts = [
        np.array([min(0.7 * emp_rate, 0.9 * rate_cap), 0.25 * beta_base, beta_base, 1.0]),
        np.array([min(0.5 * emp_rate, 0.9 * rate_cap), 0.45 * beta_base, 1.5 * beta_base, 1.5]),
        np.array([min(0.35 * emp_rate, 0.9 * rate_cap), 0.60 * beta_base, 2.5 * beta_base, 2.0]),
        np.array([min(mu0_h0, 0.95 * rate_cap), 0.08 * beta_base, 0.6 * beta_base, 0.5]),
    ]
    for x0 in starts:
        res1 = minimize(
            lambda th: -_hawkes_epoch_loglik(th, h_prev, jumps, active, dt, consts.h_crit),
            x0=x0,
            method="L-BFGS-B",
            bounds=[(1e-5, rate_cap), (0.0, beta_cap), (0.05, beta_cap), (0.0, 25.0)],
        )
        if best_h1_res is None or res1.fun < best_h1_res.fun:
            best_h1_res = res1

    mu0_h1, alpha0_h1, beta_h1, gamma_h1 = [float(v) for v in best_h1_res.x]
    ll_h1 = -float(best_h1_res.fun)

    if ll_h1 < ll_h0:
        ll_h1 = ll_h0
        mu0_h1, alpha0_h1, beta_h1, gamma_h1 = mu0_h0, 0.0, 0.85, gamma_h0

    ll_gain = ll_h1 - ll_h0
    lrt_stat = 2.0 * max(0.0, ll_gain)

    if lrt_stat <= 1e-12:
        p_val = 1.0
    else:
        p_val = float(0.5 * chi2.sf(lrt_stat, df=1) + 0.5 * chi2.sf(lrt_stat, df=2))

    df_diff = 2
    delta_bic = float(lrt_stat - df_diff * np.log(max(n_obs, 2)))

    reject_h0 = bool((p_val < significance_level) and (delta_bic > min_delta_bic) and (alpha0_h1 > 1e-4))
    active_regime = "NH-Hawkes" if reject_h0 else "NHPP-Fallback"

    ht = HypothesisTestResult(
        loglik_h0=ll_h0,
        loglik_h1=ll_h1,
        loglik_gain=ll_gain,
        lrt_statistic=lrt_stat,
        p_value=p_val,
        delta_bic=delta_bic,
        n_active_epochs=n_obs,
        reject_h0=reject_h0,
        active_regime=active_regime,
        nhpp_mu0=mu0_h0,
        nhpp_gamma_trend=gamma_h0,
    )
    return (mu0_h1, alpha0_h1, beta_h1, gamma_h1), ht


def _full_joint_loglik(
    theta_vec: np.ndarray,
    dataset: MacroEpochDataset,
    consts: StructuralConstants,
) -> float:
    """Full 8-parameter joint log-likelihood L_cont + L_point + L_jump."""
    a0, sigma0, mu0, alpha0, beta, gamma, mu_j, sigma_j = [float(x) for x in theta_vec]
    if a0 <= 0 or sigma0 <= 0 or mu0 <= 0 or alpha0 < 0 or beta <= 0 or gamma < 0 or sigma_j <= 0:
        return -1e18

    dt = dataset.dt
    h_prev = dataset.h_paths[:, :-1]
    dh = dataset.h_paths[:, 1:] - h_prev
    non_jump = dataset.active_mask & (dataset.jump_indicators == 0)

    if np.any(non_jump):
        h_sub = np.clip(h_prev[non_jump], 0.0, consts.h_crit) / consts.h_crit
        dh_sub = dh[non_jump]
        a_k = a0 * (1.0 + consts.zeta_a * (h_sub ** consts.theta_a))
        sig_k_sq = (sigma0 ** 2) * (1.0 + consts.zeta_sigma * h_sub)
        ll_cont = float(-0.5 * np.sum(np.log(2.0 * np.pi * sig_k_sq * dt) + ((dh_sub - a_k * dt) ** 2) / (sig_k_sq * dt)))
    else:
        ll_cont = 0.0

    ll_point = _hawkes_epoch_loglik(
        np.array([mu0, alpha0, beta, gamma], dtype=np.float64),
        h_prev,
        dataset.jump_indicators.astype(np.float64),
        dataset.active_mask,
        dt,
        consts.h_crit,
    )

    y = dataset.jump_sizes
    if len(y) > 0:
        ll_jump = float(-np.sum(np.log(y * sigma_j * np.sqrt(2.0 * np.pi)) + ((np.log(y) - mu_j) ** 2) / (2.0 * (sigma_j ** 2))))
    else:
        ll_jump = 0.0

    return ll_cont + ll_point + ll_jump


def _fisher_covariance(
    theta_vec: np.ndarray,
    dataset: MacroEpochDataset,
    consts: StructuralConstants,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute inverse observed Fisher Information matrix I^{-1}(theta_hat) via central finite differences."""
    d = len(theta_vec)
    eps = np.maximum(np.abs(theta_vec) * 1e-3, np.array([1e-6, 1e-5, 1e-4, 1e-4, 1e-4, 1e-4, 1e-3, 1e-3]))
    hess = np.zeros((d, d), dtype=np.float64)
    f0 = -_full_joint_loglik(theta_vec, dataset, consts)

    for i in range(d):
        ei = np.zeros(d)
        ei[i] = eps[i]
        fp = -_full_joint_loglik(theta_vec + ei, dataset, consts)
        fm = -_full_joint_loglik(theta_vec - ei, dataset, consts)
        hess[i, i] = (fp - 2.0 * f0 + fm) / (eps[i] ** 2)
        for j in range(i + 1, d):
            ej = np.zeros(d)
            ej[j] = eps[j]
            fpp = -_full_joint_loglik(theta_vec + ei + ej, dataset, consts)
            fpm = -_full_joint_loglik(theta_vec + ei - ej, dataset, consts)
            fmp = -_full_joint_loglik(theta_vec - ei + ej, dataset, consts)
            fmm = -_full_joint_loglik(theta_vec - ei - ej, dataset, consts)
            val = (fpp - fpm - fmp + fmm) / (4.0 * eps[i] * eps[j])
            hess[i, j] = val
            hess[j, i] = val

    eigvals, eigvecs = np.linalg.eigh(hess)
    eigvals_reg = np.maximum(eigvals, 1e-4)
    cov = (eigvecs * (1.0 / eigvals_reg)) @ eigvecs.T
    se = np.sqrt(np.maximum(np.diag(cov), 1e-14))
    return cov, se


def fit_hawkes_merton_mle(
    dataset: MacroEpochDataset,
    constants: Optional[StructuralConstants] = None,
    significance_level: float = 0.01,
    min_delta_bic: float = 2.0,
) -> MLEEstimationResult:
    """Full 3-stage decoupled MLE calibration + H0/H1 hypothesis test + NHPP fallback (RESS Sec. 3.4 & WP1)."""
    consts = constants or StructuralConstants()

    a0_hat, sigma0_hat = fit_continuous_wear(dataset, constants=consts)
    (mu0_hat, alpha0_hat, beta_hat, gamma_hat), ht = fit_point_process_and_test(
        dataset,
        constants=consts,
        significance_level=significance_level,
        min_delta_bic=min_delta_bic,
    )
    mu_j_hat, sigma_j_hat = fit_lognormal_jumps(dataset.jump_sizes)

    params_h1 = HawkesMertonParams(
        a0=a0_hat,
        sigma0=sigma0_hat,
        mu0=mu0_hat,
        alpha0=alpha0_hat,
        beta=beta_hat,
        gamma=gamma_hat,
        mu_J=mu_j_hat,
        sigma_J=sigma_j_hat,
    )

    if ht.reject_h0:
        effective_params = params_h1
    else:
        effective_params = HawkesMertonParams(
            a0=a0_hat,
            sigma0=sigma0_hat,
            mu0=ht.nhpp_mu0,
            alpha0=0.0,
            beta=beta_hat,
            gamma=0.0,
            mu_J=mu_j_hat,
            sigma_J=sigma_j_hat,
        )

    vec = params_h1.to_vector()
    cov, se = _fisher_covariance(vec, dataset, consts)
    total_ll = _full_joint_loglik(vec, dataset, consts)

    return MLEEstimationResult(
        params_h1=params_h1,
        effective_params=effective_params,
        standard_errors=se,
        covariance_matrix=cov,
        hypothesis_test=ht,
        total_loglik=total_ll,
    )
