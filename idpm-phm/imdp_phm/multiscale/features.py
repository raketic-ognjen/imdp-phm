"""Fast-to-Slow Timescale Separation Pipeline (RESS Section 3.2-3.3 & Concept Note WP1)."""

from dataclasses import dataclass
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert


@dataclass
class SnapshotFeatures:
    """Extracted macro-epoch features from high-frequency vibration snapshots."""
    epoch_times: np.ndarray        # [h]
    v_max: np.ndarray              # 99.9th percentile Hilbert envelope peak [g]
    excess_z: np.ndarray           # Standardized excess shock energy (0 if <= V_th)
    shock_events: np.ndarray       # Binary indicator I(V_max,k > V_th)
    h_hat: np.ndarray              # Empirical cumulative fatigue damage index H_hat(t_k)


@dataclass(frozen=True)
class BaselineNoiseProfile:
    """Calibrated healthy-state noise threshold V_th = mu_V + z_thresh * sigma_V (RESS Eq. 15)."""
    mu_v: float
    sigma_v: float
    v_th: float
    z_thresh: float = 4.5


def extract_snapshot_shock_metric(
    signal: np.ndarray,
    fs: float = 20000.0,
    f_low: float = 2000.0,
    f_high: float = 5000.0,
    quantile: float = 0.999,
    filter_order: int = 4,
) -> float:
    """Remove low-frequency rotational/kinematic BPFO modulation via bandpass resonance filtering
    and compute the 99.9th percentile Hilbert analytic envelope peak (RESS Eqs. 12-14).
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or len(x) < 32:
        raise ValueError("Input vibration snapshot must be a 1D array with at least 32 samples.")

    nyq = 0.5 * fs
    if not (0 < f_low < f_high < nyq):
        raise ValueError(f"Invalid bandpass range ({f_low}, {f_high}) for Nyquist {nyq} Hz.")

    sos = butter(filter_order, [f_low / nyq, f_high / nyq], btype="bandpass", output="sos")
    x_filt = sosfiltfilt(sos, x)
    envelope = np.abs(hilbert(x_filt))
    return float(np.quantile(envelope, quantile))


def calibrate_baseline_noise(
    v_max_nominal: np.ndarray,
    z_thresh: float = 4.5,
) -> BaselineNoiseProfile:
    """Calibrate baseline noise threshold over nominal healthy snapshots (RESS Eq. 15)."""
    arr = np.asarray(v_max_nominal, dtype=np.float64)
    if len(arr) < 5:
        raise ValueError("Need at least 5 nominal snapshots to calibrate baseline noise.")
    mu_v = float(np.mean(arr))
    sigma_v = float(np.std(arr, ddof=1))
    if sigma_v <= 1e-12:
        sigma_v = 1e-12
    v_th = mu_v + z_thresh * sigma_v
    return BaselineNoiseProfile(mu_v=mu_v, sigma_v=sigma_v, v_th=v_th, z_thresh=z_thresh)


def build_macro_epoch_trajectory(
    v_max_series: np.ndarray,
    nominal_snapshots: int = 120,
    dt: float = 0.1667,
    h0: float = 0.10,
    kappa_d: float = 0.042,
    z_thresh: float = 4.5,
) -> tuple[SnapshotFeatures, BaselineNoiseProfile]:
    """Aggregate snapshot shock metrics onto slow macro-epochs Delta t (RESS Eqs. 15-17)."""
    v_max = np.asarray(v_max_series, dtype=np.float64)
    if len(v_max) <= nominal_snapshots:
        raise ValueError("Series length must exceed nominal_snapshots calibration window.")

    profile = calibrate_baseline_noise(v_max[:nominal_snapshots], z_thresh=z_thresh)
    shock_events = (v_max > profile.v_th).astype(np.int32)
    excess_z = np.maximum(0.0, (v_max - profile.v_th) / profile.sigma_v) * shock_events
    h_hat = h0 + kappa_d * np.cumsum(excess_z * dt)
    epoch_times = np.arange(1, len(v_max) + 1, dtype=np.float64) * dt

    features = SnapshotFeatures(
        epoch_times=epoch_times,
        v_max=v_max,
        excess_z=excess_z,
        shock_events=shock_events,
        h_hat=h_hat,
    )
    return features, profile
