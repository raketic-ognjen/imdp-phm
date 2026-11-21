"""Unit & integration tests for IMS/PRONOSTIA dataset loaders, cohort MLE, and RTC/Ljung-Box diagnostics."""

from pathlib import Path
import numpy as np

from imdp_phm.core.params import HawkesMertonParams, StructuralConstants
from imdp_phm.core.sde import CoupledHawkesMertonSDE
from imdp_phm.inference.diagnostics import (
    build_cohort_dataset,
    evaluate_point_process_diagnostics,
    ljung_box_test,
)
from imdp_phm.inference.mle import MacroEpochDataset, fit_hawkes_merton_mle
from imdp_phm.multiscale.datasets import (
    discover_pronostia_bearings,
    load_ims_bearing,
    load_pronostia_bearing,
)
from imdp_phm.multiscale.features import build_macro_epoch_trajectory


def test_ims_and_pronostia_parsers_on_synthetic_files(tmp_path: Path):
    """Verify IMS timestamped files and PRONOSTIA ',' and ';' acc_*.csv files parse cleanly."""
    rng = np.random.default_rng(123)

    ims_dir = tmp_path / "IMS" / "2nd_test"
    ims_dir.mkdir(parents=True)
    for k in range(25):
        fname = f"2004.02.12.10.{k:02d}.00"
        data = 0.05 * rng.standard_normal((512, 4))
        if k >= 20:
            data[:, 0] += 1.5 * np.sin(2 * np.pi * 3200.0 * np.arange(512) / 20000.0)
        np.savetxt(ims_dir / fname, data, delimiter="\t")

    ims_rec = load_ims_bearing(tmp_path / "IMS", "2nd_test", "B1", cache_dir=tmp_path / "cache")
    assert ims_rec.is_failed_bearing is True
    assert ims_rec.n_snapshots == 25
    assert np.sum(ims_rec.features.shock_events[20:]) >= 3

    pron_root = tmp_path / "Pronostia"
    for split, b_name, delim in [
        ("Learning_set", "Bearing1_1", ","),
        ("Test_set", "Bearing1_3", ";"),
        ("Full_Test_Set", "Bearing1_3", ";"),
    ]:
        b_dir = pron_root / split / b_name
        b_dir.mkdir(parents=True)
        n_files = 10 if split == "Test_set" else 22
        for idx in range(1, n_files + 1):
            rows = np.zeros((256, 6))
            rows[:, 4] = 0.05 * rng.standard_normal(256)
            rows[:, 5] = 0.05 * rng.standard_normal(256)
            if idx >= 18:
                rows[:, 4] += 1.8 * np.sin(2 * np.pi * 3500.0 * np.arange(256) / 25600.0)
            np.savetxt(b_dir / f"acc_{idx:05d}.csv", rows, delimiter=delim)

    discovered = discover_pronostia_bearings(pron_root)
    assert discovered["Bearing1_3"][0] == "Full_Test_Set"
    pron_rec = load_pronostia_bearing(
        "Bearing1_3",
        discovered["Bearing1_3"][1],
        source_split="Full_Test_Set",
        cache_dir=tmp_path / "cache",
    )
    assert pron_rec.n_snapshots == 22
    assert np.sum(pron_rec.features.shock_events[17:]) >= 3


def test_cohort_mle_and_rtc_ljung_box_diagnostics():
    """Verify cohort dataset stacking and Random Time Change + Ljung-Box diagnostics."""
    consts = StructuralConstants(h0=0.10, h_crit=12.0, dt=0.1667)
    p_true = HawkesMertonParams(
        a0=2.5e-3,
        sigma0=1.8e-3,
        mu0=0.30,
        alpha0=0.28,
        beta=0.95,
        gamma=1.20,
        mu_J=-2.3,
        sigma_J=0.55,
    )
    sim = CoupledHawkesMertonSDE(params=p_true, constants=consts).simulate_macro_epochs(
        n_paths=15, t_max=120.0, seed=77
    )
    ds = MacroEpochDataset.from_simulation(sim, constants=consts)

    mle = fit_hawkes_merton_mle(ds, constants=consts)
    diag = evaluate_point_process_diagnostics(ds, mle, constants=consts, seed=42)
    assert diag.n_residuals_h1 >= 15
    assert diag.ks_pvalue_h1 > diag.ks_pvalue_h0
    assert diag.ks_pvalue_h1 > 0.01
    assert diag.lb_pvalue_5_h1 > 0.01

    rng = np.random.default_rng(9)
    f1, _ = build_macro_epoch_trajectory(
        np.concatenate([0.1 + 0.01 * rng.standard_normal(50), np.full(15, 0.8)]),
        nominal_snapshots=30,
    )
    f2, _ = build_macro_epoch_trajectory(
        np.concatenate([0.1 + 0.01 * rng.standard_normal(70), np.full(20, 0.9)]),
        nominal_snapshots=30,
    )
    cohort_ds = build_cohort_dataset([f1, f2], constants=consts)
    assert cohort_ds.h_paths.shape == (2, 91)
    assert np.sum(cohort_ds.active_mask[0]) <= 65
    assert np.sum(cohort_ds.active_mask[1]) <= 90

    q_white, p_white = ljung_box_test(rng.standard_normal(200), lags=5)
    assert p_white > 0.05
