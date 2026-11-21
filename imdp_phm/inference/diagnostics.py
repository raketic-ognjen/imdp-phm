"""Extended Point-Process Goodness-of-Fit Diagnostics & Multi-Trajectory Cohort Builder (WP1)."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.stats import chi2, kstest

from imdp_phm.core.params import StructuralConstants
from imdp_phm.inference.mle import MacroEpochDataset, MLEEstimationResult
from imdp_phm.multiscale.features import SnapshotFeatures


@dataclass(frozen=True)
class PointProcessDiagnostics:
    """Goodness-of-fit diagnostics comparing H0 (NHPP) vs H1 (NH-Hawkes)
    via discrete-time Random Time Change compensator residuals and Ljung-Box independence.
    """
    n_trajectories: int
    n_active_epochs: int
    total_active_hours: float
    total_jumps: int
    empirical_jump_rate_per_hour: float
    nominal_branching_ratio: float
    max_branching_ratio: float
    n_residuals_h0: int
    residual_mean_h0: float
    residual_std_h0: float
    dispersion_index_h0: float
    ks_stat_h0: float
    ks_pvalue_h0: float
    lb_stat_5_h0: float
    lb_pvalue_5_h0: float
    n_residuals_h1: int
    residual_mean_h1: float
    residual_std_h1: float
    dispersion_index_h1: float
    ks_stat_h1: float
    ks_pvalue_h1: float
    lb_stat_5_h1: float
    lb_pvalue_5_h1: float
    lb_stat_10_h1: float
    lb_pvalue_10_h1: float


def build_cohort_dataset(
    features_list: list[SnapshotFeatures],
    constants: Optional[StructuralConstants] = None,
) -> MacroEpochDataset:
    """Stack M >= 1 variable-length SnapshotFeatures trajectories into a padded MacroEpochDataset."""
    if not features_list:
        raise ValueError("features_list must contain at least one SnapshotFeatures object.")

    consts = constants or StructuralConstants()
    m_paths = len(features_list)
    lengths = [len(f.h_hat) for f in features_list]
    k_max = max(lengths)

    dt_vals = []
    for f in features_list:
        if len(f.epoch_times) > 1:
            dt_vals.append(float(np.median(np.diff(f.epoch_times))))
        else:
            dt_vals.append(consts.dt)
    dt_cohort = float(np.median(dt_vals))

    h_paths = np.full((m_paths, k_max + 1), consts.h_crit, dtype=np.float64)
    jump_indicators = np.zeros((m_paths, k_max), dtype=np.int32)
    active_mask = np.zeros((m_paths, k_max), dtype=bool)
    all_y: list[np.ndarray] = []

    for m, feat in enumerate(features_list):
        k_m = len(feat.h_hat)
        h_seq = np.concatenate([[consts.h0], np.asarray(feat.h_hat, dtype=np.float64)])
        h_paths[m, : k_m + 1] = h_seq
        j_seq = (np.asarray(feat.shock_events) > 0).astype(np.int32)
        jump_indicators[m, :k_m] = j_seq
        act_m = h_seq[:-1] < consts.h_crit
        active_mask[m, :k_m] = act_m

        dh_m = h_seq[1:] - h_seq[:-1]
        y_m = dh_m[(j_seq > 0) & act_m]
        y_m = y_m[y_m > 1e-9]
        if len(y_m) > 0:
            all_y.append(y_m)

    y_obs = np.concatenate(all_y) if all_y else np.empty(0, dtype=np.float64)
    return MacroEpochDataset(
        dt=dt_cohort,
        h_paths=h_paths,
        jump_indicators=jump_indicators,
        jump_sizes=y_obs,
        active_mask=active_mask,
    )


def ljung_box_test(series: np.ndarray, lags: int = 5) -> tuple[float, float]:
    """Pure NumPy/SciPy Ljung-Box Q(L) test for serial autocorrelation up to lag L."""
    x = np.asarray(series, dtype=np.float64)
    n = len(x)
    if n <= lags + 2:
        return 0.0, 1.0
    x_centered = x - np.mean(x)
    denom = float(np.dot(x_centered, x_centered))
    if denom <= 1e-18:
        return 0.0, 1.0

    q_stat = 0.0
    for lag in range(1, lags + 1):
        num = float(np.dot(x_centered[lag:], x_centered[:-lag]))
        rho_l = num / denom
        q_stat += (rho_l ** 2) / (n - lag)

    q_stat *= n * (n + 2)
    p_val = float(chi2.sf(q_stat, df=lags))
    return float(q_stat), p_val


def _reconstruct_intensity_paths(
    dataset: MacroEpochDataset,
    mle_result: MLEEstimationResult,
    constants: StructuralConstants,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct discrete-time conditional intensity matrices lambda_h0[m, k] and lambda_h1[m, k]."""
    h_prev = dataset.h_paths[:, :-1]
    h_norm_sq = (np.clip(h_prev, 0.0, constants.h_crit) / constants.h_crit) ** 2

    ht = mle_result.hypothesis_test
    lam_h0 = ht.nhpp_mu0 * (1.0 + ht.nhpp_gamma_trend * h_norm_sq)

    p1 = mle_result.params_h1
    m_paths, k_steps = dataset.jump_indicators.shape
    lam_h1 = np.zeros((m_paths, k_steps), dtype=np.float64)
    lam_curr = np.full(m_paths, p1.mu0, dtype=np.float64)
    rho = float(np.exp(-p1.beta * dataset.dt))
    alpha_mat = p1.alpha0 * (1.0 + p1.gamma * h_norm_sq)
    jumps = dataset.jump_indicators.astype(np.float64)

    for k in range(k_steps):
        lam_h1[:, k] = lam_curr
        lam_curr = p1.mu0 + (lam_curr - p1.mu0) * rho + jumps[:, k] * alpha_mat[:, k]

    return lam_h0, lam_h1


def _discrete_compensator_residuals(
    lam_matrix: np.ndarray,
    jump_indicators: np.ndarray,
    active_mask: np.ndarray,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute randomized discrete-time Random Time Change residuals tau_r ~ Exp(1)."""
    residuals: list[float] = []
    m_paths = jump_indicators.shape[0]
    for m in range(m_paths):
        act = active_mask[m]
        k_end = int(np.sum(act))
        if k_end == 0:
            continue
        lam_m = lam_matrix[m, :k_end]
        jmp_m = jump_indicators[m, :k_end] > 0
        jump_idx = np.flatnonzero(jmp_m)
        if len(jump_idx) < 2:
            continue

        prev_k = -1
        for k_j in jump_idx:
            u_jitter = float(rng.uniform(0.0, 1.0))
            p_j = float(np.clip(1.0 - np.exp(-float(lam_m[k_j]) * dt), 1e-15, 1.0 - 1e-15))
            intra_tau = -float(np.log(1.0 - u_jitter * p_j))
            if k_j == prev_k + 1:
                tau_r = intra_tau
            else:
                full_integral = float(np.sum(lam_m[prev_k + 1 : k_j])) * dt
                tau_r = full_integral + intra_tau
            residuals.append(max(tau_r, 1e-12))
            prev_k = int(k_j)

    return np.asarray(residuals, dtype=np.float64)


def evaluate_point_process_diagnostics(
    dataset: MacroEpochDataset,
    mle_result: MLEEstimationResult,
    constants: Optional[StructuralConstants] = None,
    seed: int = 42,
) -> PointProcessDiagnostics:
    """Evaluate Random Time Change Exp(1) KS goodness-of-fit and Ljung-Box independence."""
    consts = constants or StructuralConstants()
    rng = np.random.Generator(np.random.PCG64(seed))

    lam_h0, lam_h1 = _reconstruct_intensity_paths(dataset, mle_result, consts)
    res_h0 = _discrete_compensator_residuals(
        lam_h0, dataset.jump_indicators, dataset.active_mask, dataset.dt, rng
    )
    res_h1 = _discrete_compensator_residuals(
        lam_h1, dataset.jump_indicators, dataset.active_mask, dataset.dt, rng
    )

    def _summarize(res: np.ndarray) -> tuple[int, float, float, float, float, float, float, float, float, float]:
        n_r = len(res)
        if n_r < 5:
            return n_r, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0
        mean_r = float(np.mean(res))
        std_r = float(np.std(res, ddof=1))
        disp_r = float((std_r ** 2) / max(mean_r, 1e-12))
        ks_d, ks_p = kstest(res, "expon", args=(0.0, 1.0))
        u_pit = 1.0 - np.exp(-res)
        lb5_q, lb5_p = ljung_box_test(u_pit, lags=5)
        lb10_q, lb10_p = ljung_box_test(u_pit, lags=10)
        return n_r, mean_r, std_r, disp_r, float(ks_d), float(ks_p), lb5_q, lb5_p, lb10_q, lb10_p

    n0, m0, s0, disp0, ks_d0, ks_p0, lb5_q0, lb5_p0, _, _ = _summarize(res_h0)
    n1, m1, s1, disp1, ks_d1, ks_p1, lb5_q1, lb5_p1, lb10_q1, lb10_p1 = _summarize(res_h1)

    n_act = int(np.sum(dataset.active_mask))
    tot_hours = float(n_act * dataset.dt)
    tot_jumps = int(np.sum((dataset.jump_indicators > 0) & dataset.active_mask))
    emp_rate = float(tot_jumps / max(tot_hours, 1e-9))

    p1 = mle_result.params_h1
    return PointProcessDiagnostics(
        n_trajectories=int(dataset.h_paths.shape[0]),
        n_active_epochs=n_act,
        total_active_hours=tot_hours,
        total_jumps=tot_jumps,
        empirical_jump_rate_per_hour=emp_rate,
        nominal_branching_ratio=float(p1.nominal_branching_ratio()),
        max_branching_ratio=float(p1.max_branching_ratio()),
        n_residuals_h0=n0,
        residual_mean_h0=m0,
        residual_std_h0=s0,
        dispersion_index_h0=disp0,
        ks_stat_h0=ks_d0,
        ks_pvalue_h0=ks_p0,
        lb_stat_5_h0=lb5_q0,
        lb_pvalue_5_h0=lb5_p0,
        n_residuals_h1=n1,
        residual_mean_h1=m1,
        residual_std_h1=s1,
        dispersion_index_h1=disp1,
        ks_stat_h1=ks_d1,
        ks_pvalue_h1=ks_p1,
        lb_stat_5_h1=lb5_q1,
        lb_pvalue_5_h1=lb5_p1,
        lb_stat_10_h1=lb10_q1,
        lb_pvalue_10_h1=lb10_p1,
    )


def reconstruct_intensity_paths(
    dataset: MacroEpochDataset,
    mle_result: MLEEstimationResult,
    constants: Optional[StructuralConstants] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Public wrapper returning discrete-time (lambda_h0, lambda_h1) matrices."""
    consts = constants or StructuralConstants()
    return _reconstruct_intensity_paths(dataset, mle_result, consts)
