"""Empirical Bearing Dataset Loaders & Fast-to-Slow Feature Adapters for NASA IMS and PRONOSTIA (FEMTO-ST)."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Optional
import numpy as np

from imdp_phm.core.params import StructuralConstants
from imdp_phm.multiscale.features import (
    BaselineNoiseProfile,
    SnapshotFeatures,
    calibrate_baseline_noise,
    extract_snapshot_shock_metric,
)

_IMS_TIMESTAMP_RE = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$")
_PRONOSTIA_ACC_RE = re.compile(r"^acc_(\d+)\.csv$", re.IGNORECASE)

IMS_BEARING_CATALOG: dict[str, dict[str, tuple[int, bool, str]]] = {
    "1st_test": {
        "B1": (0, False, "Healthy control"),
        "B2": (2, False, "Healthy control"),
        "B3": (4, True, "Inner race defect"),
        "B4": (6, True, "Roller element defect"),
    },
    "2nd_test": {
        "B1": (0, True, "Outer race failure"),
        "B2": (1, False, "Healthy control"),
        "B3": (2, False, "Healthy control"),
        "B4": (3, False, "Healthy control"),
    },
    "3rd_test": {
        "B1": (0, False, "Healthy control"),
        "B2": (1, False, "Healthy control"),
        "B3": (2, True, "Outer race failure"),
        "B4": (3, False, "Healthy control"),
    },
}

PRONOSTIA_CONDITIONS: dict[str, str] = {
    "1": "1800 rpm / 4000 N",
    "2": "1650 rpm / 4200 N",
    "3": "1500 rpm / 5000 N",
}


@dataclass
class BearingTrajectoryRecord:
    """Loaded and feature-extracted run-to-failure (or healthy control) bearing record."""
    dataset_name: str
    subset_name: str
    bearing_id: str
    condition_label: str
    is_failed_bearing: bool
    sampling_rate_hz: float
    dt_hours: float
    duration_hours: float
    n_snapshots: int
    features: SnapshotFeatures
    noise_profile: BaselineNoiseProfile


def resolve_desktop_dir(folder_name: str) -> Path:
    """Resolve ~/Desktop/<folder_name> across standard Linux/macOS/Windows layouts."""
    home = Path.home()
    candidates = [
        home / "Desktop" / folder_name,
        home / "OneDrive" / "Desktop" / folder_name,
        Path(folder_name),
    ]
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    raise FileNotFoundError(
        f"Could not locate '{folder_name}' folder. Checked: {[str(p) for p in candidates]}"
    )


def build_variable_dt_trajectory(
    v_max_series: np.ndarray,
    epoch_times_hours: np.ndarray,
    nominal_fraction: float = 0.15,
    min_nominal: int = 30,
    max_nominal: int = 120,
    h0: float = 0.10,
    kappa_d: float = 0.042,
    z_thresh: float = 4.5,
    max_step_hours: float = 0.50,
) -> tuple[SnapshotFeatures, BaselineNoiseProfile]:
    """Aggregate snapshot shock metrics onto macro-epochs using actual elapsed timestamps."""
    v_max = np.asarray(v_max_series, dtype=np.float64)
    t_hrs = np.asarray(epoch_times_hours, dtype=np.float64)
    if len(v_max) != len(t_hrs):
        raise ValueError("v_max_series and epoch_times_hours must have identical length.")
    if len(v_max) < 15:
        raise ValueError(f"Trajectory too short ({len(v_max)} snapshots); need at least 15.")

    n_nom = int(np.clip(np.floor(nominal_fraction * len(v_max)), min_nominal, min(max_nominal, len(v_max) // 2)))
    n_nom = max(5, n_nom)

    profile = calibrate_baseline_noise(v_max[:n_nom], z_thresh=z_thresh)
    shock_events = (v_max > profile.v_th).astype(np.int32)
    excess_z = np.maximum(0.0, (v_max - profile.v_th) / profile.sigma_v) * shock_events

    raw_dt = np.diff(np.concatenate([[0.0], t_hrs]))
    median_dt = float(np.median(raw_dt[raw_dt > 0])) if np.any(raw_dt > 0) else 0.1667
    effective_dt = np.clip(np.where(raw_dt > 0, raw_dt, median_dt), 1e-5, max_step_hours)

    h_hat = h0 + kappa_d * np.cumsum(excess_z * effective_dt)
    clean_epoch_times = np.cumsum(effective_dt)

    features = SnapshotFeatures(
        epoch_times=clean_epoch_times,
        v_max=v_max,
        excess_z=excess_z,
        shock_events=shock_events,
        h_hat=h_hat,
    )
    return features, profile


def _find_ims_files(test_dir: Path) -> list[Path]:
    """Find all timestamp-named snapshot files inside an IMS test directory."""
    files = [
        p for p in test_dir.rglob("*")
        if p.is_file() and _IMS_TIMESTAMP_RE.match(p.name)
    ]
    files.sort(key=lambda p: datetime.strptime(p.name, "%Y.%m.%d.%H.%M.%S"))
    return files


def load_ims_bearing(
    ims_root: Path,
    test_name: str,
    bearing_code: str,
    constants: Optional[StructuralConstants] = None,
    z_thresh: Optional[float] = None,
    cache_dir: Optional[Path] = None,
) -> BearingTrajectoryRecord:
    """Load one NASA IMS bearing channel, extract 2-5 kHz Hilbert envelope shock series (cached)."""
    consts = constants or StructuralConstants()
    z_val = consts.z_thresh if z_thresh is None else float(z_thresh)
    if test_name not in IMS_BEARING_CATALOG or bearing_code not in IMS_BEARING_CATALOG[test_name]:
        raise ValueError(f"Unknown IMS target ({test_name}, {bearing_code}).")

    col_idx, is_failed, desc = IMS_BEARING_CATALOG[test_name][bearing_code]
    bearing_id = f"IMS_{test_name}_{bearing_code}"
    fs = 20000.0

    cache_file = (cache_dir / f"{bearing_id}.npz") if cache_dir is not None else None
    if cache_file is not None and cache_file.exists():
        data = np.load(cache_file)
        v_max_series = data["v_max"]
        epoch_times = data["epoch_times"]
    else:
        test_dir = ims_root / test_name
        if not test_dir.exists():
            raise FileNotFoundError(f"IMS test directory not found: {test_dir}")
        files = _find_ims_files(test_dir)
        if not files:
            raise FileNotFoundError(f"No timestamped IMS snapshot files found under {test_dir}")

        t0 = datetime.strptime(files[0].name, "%Y.%m.%d.%H.%M.%S")
        v_list: list[float] = []
        t_list: list[float] = []

        for idx, fp in enumerate(files):
            dt_obj = datetime.strptime(fp.name, "%Y.%m.%d.%H.%M.%S")
            elapsed_h = (dt_obj - t0).total_seconds() / 3600.0
            if idx == 0:
                elapsed_h = 10.0 / 60.0
            raw = np.loadtxt(fp, dtype=np.float64)
            sig = raw if raw.ndim == 1 else raw[:, col_idx]
            v_list.append(extract_snapshot_shock_metric(sig, fs=fs))
            t_list.append(elapsed_h)

        v_max_series = np.asarray(v_list, dtype=np.float64)
        epoch_times = np.asarray(t_list, dtype=np.float64)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_file, v_max=v_max_series, epoch_times=epoch_times)

    features, profile = build_variable_dt_trajectory(
        v_max_series=v_max_series,
        epoch_times_hours=epoch_times,
        h0=consts.h0,
        kappa_d=consts.kappa_d,
        z_thresh=z_val,
        max_step_hours=0.50,
    )
    dt_med = float(np.median(np.diff(np.concatenate([[0.0], features.epoch_times]))))
    return BearingTrajectoryRecord(
        dataset_name="NASA_IMS",
        subset_name=test_name,
        bearing_id=bearing_id,
        condition_label=desc,
        is_failed_bearing=is_failed,
        sampling_rate_hz=fs,
        dt_hours=dt_med,
        duration_hours=float(features.epoch_times[-1]),
        n_snapshots=len(v_max_series),
        features=features,
        noise_profile=profile,
    )


def _read_pronostia_acc_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a PRONOSTIA acc_XXXXX.csv file auto-detecting ',' vs ';' delimiter."""
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    delim = ";" if ";" in first_line else ","
    arr = np.loadtxt(csv_path, delimiter=delim, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"Unexpected PRONOSTIA CSV shape {arr.shape} in {csv_path}")
    return arr[:, 4], arr[:, 5]


def discover_pronostia_bearings(pronostia_root: Path) -> dict[str, tuple[str, Path]]:
    """Discover all unique PRONOSTIA bearings prioritizing Full_Test_Set > Learning_set > Test_set."""
    priority_folders = ["Full_Test_Set", "Learning_set", "Test_set"]
    discovered: dict[str, tuple[str, Path]] = {}
    for folder in priority_folders:
        sub = pronostia_root / folder
        if not sub.exists() or not sub.is_dir():
            continue
        for b_dir in sorted(sub.iterdir()):
            if b_dir.is_dir() and b_dir.name.lower().startswith("bearing"):
                if b_dir.name not in discovered:
                    discovered[b_dir.name] = (folder, b_dir)
    return dict(sorted(discovered.items()))


def load_pronostia_bearing(
    bearing_name: str,
    bearing_dir: Path,
    source_split: str = "Full_Test_Set",
    channel_mode: str = "max_hv",
    constants: Optional[StructuralConstants] = None,
    z_thresh: Optional[float] = None,
    cache_dir: Optional[Path] = None,
) -> BearingTrajectoryRecord:
    """Load one PRONOSTIA (FEMTO-ST) bearing trajectory from acc_XXXXX.csv files (cached)."""
    consts = constants or StructuralConstants()
    z_val = consts.z_thresh if z_thresh is None else float(z_thresh)
    fs = 25600.0
    dt_fixed = 10.0 / 3600.0
    bearing_id = f"PRONOSTIA_{bearing_name}"

    cond_digit = bearing_name.replace("Bearing", "").split("_")[0]
    cond_str = PRONOSTIA_CONDITIONS.get(cond_digit, "Unknown condition")

    cache_file = (cache_dir / f"{bearing_id}_{channel_mode}.npz") if cache_dir is not None else None
    if cache_file is not None and cache_file.exists():
        data = np.load(cache_file)
        v_max_series = data["v_max"]
        epoch_times = data["epoch_times"]
    else:
        acc_files = [
            p for p in bearing_dir.iterdir()
            if p.is_file() and _PRONOSTIA_ACC_RE.match(p.name)
        ]
        if not acc_files:
            raise FileNotFoundError(f"No acc_*.csv files found in {bearing_dir}")
        acc_files.sort(key=lambda p: int(_PRONOSTIA_ACC_RE.match(p.name).group(1)))

        v_list: list[float] = []
        for fp in acc_files:
            horiz, vert = _read_pronostia_acc_csv(fp)
            if channel_mode == "horiz":
                v_val = extract_snapshot_shock_metric(horiz, fs=fs)
            elif channel_mode == "vert":
                v_val = extract_snapshot_shock_metric(vert, fs=fs)
            else:
                vh = extract_snapshot_shock_metric(horiz, fs=fs)
                vv = extract_snapshot_shock_metric(vert, fs=fs)
                v_val = max(vh, vv)
            v_list.append(v_val)

        v_max_series = np.asarray(v_list, dtype=np.float64)
        epoch_times = np.arange(1, len(v_max_series) + 1, dtype=np.float64) * dt_fixed
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_file, v_max=v_max_series, epoch_times=epoch_times)

    scaled_kappa_d = 0.0065 * (0.1667 / dt_fixed)
    features, profile = build_variable_dt_trajectory(
        v_max_series=v_max_series,
        epoch_times_hours=epoch_times,
        h0=consts.h0,
        kappa_d=scaled_kappa_d,
        z_thresh=z_val,
        max_step_hours=0.10,
    )
    return BearingTrajectoryRecord(
        dataset_name="PRONOSTIA",
        subset_name=f"Condition_{cond_digit} ({source_split})",
        bearing_id=bearing_id,
        condition_label=cond_str,
        is_failed_bearing=True,
        sampling_rate_hz=fs,
        dt_hours=dt_fixed,
        duration_hours=float(features.epoch_times[-1]),
        n_snapshots=len(v_max_series),
        features=features,
        noise_profile=profile,
    )
