"""End-to-End Empirical WP1 + WP2 Validation Runner on NASA IMS & PRONOSTIA (FEMTO-ST) Datasets."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np

from imdp_phm.core.params import StructuralConstants
from imdp_phm.inference.credal import CredalParameterSet
from imdp_phm.inference.diagnostics import (
    build_cohort_dataset,
    evaluate_point_process_diagnostics,
)
from imdp_phm.inference.mle import MacroEpochDataset, fit_hawkes_merton_mle
from imdp_phm.multiscale.datasets import (
    IMS_BEARING_CATALOG,
    BearingTrajectoryRecord,
    discover_pronostia_bearings,
    load_ims_bearing,
    load_pronostia_bearing,
    resolve_desktop_dir,
)
from imdp_phm.solver.bellman import solve_imdp_minimax
from imdp_phm.solver.grid import MaintenanceCostModel, StateGrid2D
from imdp_phm.solver.transition import solve_point_mdp


def evaluate_record(
    label: str,
    dataset: MacroEpochDataset,
    consts: StructuralConstants,
    run_wp2: bool = False,
) -> dict:
    mle_res = fit_hawkes_merton_mle(dataset, constants=consts, significance_level=0.01, min_delta_bic=2.0)
    diag = evaluate_point_process_diagnostics(dataset, mle_res, constants=consts, seed=42)
    ht = mle_res.hypothesis_test
    p1 = mle_res.params_h1

    row = {
        "cohort_or_bearing": label,
        "n_trajectories": diag.n_trajectories,
        "active_epochs": diag.n_active_epochs,
        "active_hours": round(diag.total_active_hours, 2),
        "jumps": diag.total_jumps,
        "regime": ht.active_regime,
        "reject_h0": ht.reject_h0,
        "lrt_stat": round(ht.lrt_statistic, 3),
        "lrt_pvalue": float(ht.p_value),
        "delta_bic": round(ht.delta_bic, 3),
        "alpha0": round(p1.alpha0, 4),
        "beta": round(p1.beta, 4),
        "gamma": round(p1.gamma, 4),
        "eta_0": round(diag.nominal_branching_ratio, 4),
        "eta_max": round(diag.max_branching_ratio, 4),
        "ks_stat_h0": round(diag.ks_stat_h0, 4),
        "ks_pval_h0": round(diag.ks_pvalue_h0, 4),
        "ks_stat_h1": round(diag.ks_stat_h1, 4),
        "ks_pval_h1": round(diag.ks_pvalue_h1, 4),
        "lb5_pval_h0": round(diag.lb_pvalue_5_h0, 4),
        "lb5_pval_h1": round(diag.lb_pvalue_5_h1, 4),
        "lb10_pval_h1": round(diag.lb_pvalue_10_h1, 4),
        "disp_h0": round(diag.dispersion_index_h0, 3),
        "disp_h1": round(diag.dispersion_index_h1, 3),
        "mle_params_h1": asdict(p1),
    }

    if run_wp2:
        credal = CredalParameterSet.from_mle_result(mle_res, n_trajectories=diag.n_trajectories, z_mult=1.96)
        grid = StateGrid2D.uniform(n_h=35, n_lam=21, lam_min=0.05, lam_max=2.50, constants=consts)
        cm = MaintenanceCostModel()
        imdp_res = solve_imdp_minimax(grid=grid, credal_set=credal, cost_model=cm, constants=consts, tol=1e-4, seed=42)
        point_res = solve_point_mdp(grid=grid, params=mle_res.effective_params, cost_model=cm, constants=consts, tol=1e-5)
        v_imdp = imdp_res.value_at(grid, consts.h0, mle_res.effective_params.mu0)
        v_mle = point_res.value_at(grid, consts.h0, mle_res.effective_params.mu0)
        row["wp2_imdp_pristine_cost"] = round(v_imdp, 2)
        row["wp2_mle_pristine_cost"] = round(v_mle, 2)
        row["wp2_epistemic_margin"] = round(v_imdp - v_mle, 2)
        row["wp2_policy_regions"] = imdp_res.policy_region_fractions()

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run empirical WP1+WP2 validation on NASA IMS & PRONOSTIA.")
    parser.add_argument("--ims-dir", type=Path, default=None, help="Path to IMS folder (default: ~/Desktop/IMS)")
    parser.add_argument("--pronostia-dir", type=Path, default=None, help="Path to Pronostia folder (default: ~/Desktop/Pronostia)")
    parser.add_argument("--out-dir", type=Path, default=Path("results"), help="Output directory for cache & reports")
    parser.add_argument("--z-thresh", type=float, default=4.0, help="Primary shock detection threshold z_thresh")
    args = parser.parse_args()

    t0 = time.perf_counter()
    ims_root = args.ims_dir or resolve_desktop_dir("IMS")
    pron_root = args.pronostia_dir or resolve_desktop_dir("Pronostia")
    cache_dir = args.out_dir / "cache"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    consts = StructuralConstants(z_thresh=args.z_thresh)
    print(f"[1/4] Loading NASA IMS trajectories from {ims_root} (cached in {cache_dir})...")
    ims_records: list[BearingTrajectoryRecord] = []
    for test_name, bearings in IMS_BEARING_CATALOG.items():
        for b_code in bearings:
            rec = load_ims_bearing(
                ims_root=ims_root,
                test_name=test_name,
                bearing_code=b_code,
                constants=consts,
                z_thresh=args.z_thresh,
                cache_dir=cache_dir,
            )
            ims_records.append(rec)
            print(
                f"  - {rec.bearing_id:18s} | failed={str(rec.is_failed_bearing):5s} | "
                f"n={rec.n_snapshots:4d} | jumps={int(np.sum(rec.features.shock_events)):3d} | "
                f"H_end={rec.features.h_hat[-1]:.3f}"
            )

    print(f"\n[2/4] Loading PRONOSTIA trajectories from {pron_root} (cached in {cache_dir})...")
    pron_map = discover_pronostia_bearings(pron_root)
    pron_records: list[BearingTrajectoryRecord] = []
    for b_name, (split_name, b_dir) in pron_map.items():
        rec = load_pronostia_bearing(
            bearing_name=b_name,
            bearing_dir=b_dir,
            source_split=split_name,
            channel_mode="max_hv",
            constants=consts,
            z_thresh=args.z_thresh,
            cache_dir=cache_dir,
        )
        pron_records.append(rec)
        print(
            f"  - {rec.bearing_id:20s} | {rec.subset_name:28s} | "
            f"n={rec.n_snapshots:4d} | jumps={int(np.sum(rec.features.shock_events)):3d} | "
            f"H_end={rec.features.h_hat[-1]:.3f}"
        )

    print("\n[3/4] Running WP1 Point-Process MLE, LRT, Delta-BIC, KS & Ljung-Box Diagnostics...")
    results: list[dict] = []

    ims_failed = [r for r in ims_records if r.is_failed_bearing]
    ims_healthy = [r for r in ims_records if not r.is_failed_bearing]
    for r in ims_failed:
        ds = MacroEpochDataset.from_features(r.features, constants=consts)
        results.append(evaluate_record(r.bearing_id, ds, consts, run_wp2=True))

    results.append(
        evaluate_record(
            "COHORT_IMS_FAILED (4 bearings)",
            build_cohort_dataset([r.features for r in ims_failed], constants=consts),
            consts,
            run_wp2=True,
        )
    )
    results.append(
        evaluate_record(
            "CONTROL_IMS_HEALTHY (8 bearings)",
            build_cohort_dataset([r.features for r in ims_healthy], constants=consts),
            consts,
            run_wp2=False,
        )
    )

    for r in pron_records:
        ds = MacroEpochDataset.from_features(r.features, constants=consts)
        results.append(evaluate_record(r.bearing_id, ds, consts, run_wp2=False))

    for cond_idx in ("1", "2", "3"):
        c_recs = [r for r in pron_records if r.bearing_id.startswith(f"PRONOSTIA_Bearing{cond_idx}_")]
        if c_recs:
            results.append(
                evaluate_record(
                    f"COHORT_PRONOSTIA_COND_{cond_idx} ({len(c_recs)} bearings)",
                    build_cohort_dataset([r.features for r in c_recs], constants=consts),
                    consts,
                    run_wp2=True,
                )
            )

    results.append(
        evaluate_record(
            f"COHORT_PRONOSTIA_ALL ({len(pron_records)} bearings)",
            build_cohort_dataset([r.features for r in pron_records], constants=consts),
            consts,
            run_wp2=True,
        )
    )

    sensitivity_rows: list[dict] = []
    for z_sens in (3.5, 4.0, 4.5):
        c_sens = StructuralConstants(z_thresh=z_sens)
        ims_f_sens = [
            load_ims_bearing(ims_root, r.subset_name, r.bearing_id.split("_")[-1], constants=c_sens, z_thresh=z_sens, cache_dir=cache_dir)
            for r in ims_failed
        ]
        pron_sens = [
            load_pronostia_bearing(r.bearing_id.replace("PRONOSTIA_", ""), pron_map[r.bearing_id.replace("PRONOSTIA_", "")][1], constants=c_sens, z_thresh=z_sens, cache_dir=cache_dir)
            for r in pron_records
        ]
        sensitivity_rows.append(
            evaluate_record(
                f"SENS_IMS_FAILED_z={z_sens}",
                build_cohort_dataset([x.features for x in ims_f_sens], constants=c_sens),
                c_sens,
                run_wp2=False,
            )
        )
        sensitivity_rows.append(
            evaluate_record(
                f"SENS_PRONOSTIA_ALL_z={z_sens}",
                build_cohort_dataset([x.features for x in pron_sens], constants=c_sens),
                c_sens,
                run_wp2=False,
            )
        )

    print("\n[4/4] Writing Markdown & JSON reports...")
    md_lines = [
        f"# Empirical WP1 & WP2 Validation Report (`z_thresh = {args.z_thresh}`)",
        "",
        "| Target | M | Jumps | Regime | LRT | p-val | ΔBIC | η₀ | η_max | KS p (H₀ → H₁) | LB(5) p (H₀ → H₁) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in results:
        md_lines.append(
            f"| {r['cohort_or_bearing']} | {r['n_trajectories']} | {r['jumps']} | **{r['regime']}** | "
            f"{r['lrt_stat']:.2f} | {r['lrt_pvalue']:.2e} | {r['delta_bic']:.2f} | "
            f"{r['eta_0']:.3f} | {r['eta_max']:.3f} | "
            f"{r['ks_pval_h0']:.3f} → **{r['ks_pval_h1']:.3f}** | "
            f"{r['lb5_pval_h0']:.3f} → **{r['lb5_pval_h1']:.3f}** |"
        )

    md_lines.extend([
        "",
        "## Threshold Sensitivity Sweep (`z_thresh` ∈ {3.5, 4.0, 4.5})",
        "",
        "| Cohort & Threshold | M | Jumps | Regime | LRT | p-val | ΔBIC | η₀ | η_max | KS p (H₁) | LB(5) p (H₁) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in sensitivity_rows:
        md_lines.append(
            f"| {r['cohort_or_bearing']} | {r['n_trajectories']} | {r['jumps']} | **{r['regime']}** | "
            f"{r['lrt_stat']:.2f} | {r['lrt_pvalue']:.2e} | {r['delta_bic']:.2f} | "
            f"{r['eta_0']:.3f} | {r['eta_max']:.3f} | {r['ks_pval_h1']:.3f} | {r['lb5_pval_h1']:.3f} |"
        )

    report_md = "\n".join(md_lines)
    md_path = args.out_dir / "empirical_validation_report.md"
    json_path = args.out_dir / "empirical_validation_results.json"
    md_path.write_text(report_md, encoding="utf-8")
    json_path.write_text(
        json.dumps({"primary": results, "sensitivity": sensitivity_rows}, indent=2),
        encoding="utf-8",
    )

    print("\n" + report_md)
    print(f"\nCompleted in {time.perf_counter() - t0:.2f}s. Saved to {md_path} and {json_path}.")


if __name__ == "__main__":
    main()
