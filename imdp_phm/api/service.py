import torch
torch.set_num_threads(1)
"""Stateless Service Layer connecting FastAPI routes to WP1, WP2, WP3, and Empirical engines."""

from pathlib import Path
import time
from typing import Optional
import numpy as np
from scipy.stats import norm

from imdp_phm.api.schemas import (
    BenchmarkRequest,
    BenchmarkResponse,
    ConstantsSchema,
    CostModelSchema,
    CredalIntervalItem,
    DefaultsResponse,
    EmpiricalCatalogItem,
    EmpiricalCatalogResponse,
    EmpiricalDiagnosticsSummary,
    EmpiricalRunRequest,
    EmpiricalRunResponse,
    EmpiricalTrajectorySeries,
    FullStudyRequest,
    FullStudyResponse,
    HealthResponse,
    HypothesisSummary,
    InferenceRequest,
    InferenceResponse,
    JumpPBoxSummary,
    ParamsSchema,
    PolicyBenchmarkRow,
    SimulationRequest,
    SimulationResponse,
    SolverRequest,
    SolverResponse,
    TrajectoryPreview,
)
from imdp_phm.benchmark.evaluator import run_wp3_competitive_benchmark
from imdp_phm.core.params import HawkesMertonParams, StructuralConstants
from imdp_phm.core.sde import CoupledHawkesMertonSDE
from imdp_phm.inference.credal import PARAM_NAMES, CredalParameterSet
from imdp_phm.inference.diagnostics import (
    build_cohort_dataset,
    evaluate_point_process_diagnostics,
    reconstruct_intensity_paths,
)
from imdp_phm.inference.mle import MacroEpochDataset, fit_hawkes_merton_mle
from imdp_phm.multiscale.datasets import (
    IMS_BEARING_CATALOG,
    PRONOSTIA_CONDITIONS,
    BearingTrajectoryRecord,
    discover_pronostia_bearings,
    load_ims_bearing,
    load_pronostia_bearing,
    resolve_desktop_dir,
)
from imdp_phm.solver.bellman import solve_imdp_minimax
from imdp_phm.solver.grid import MaintenanceCostModel, StateGrid2D
from imdp_phm.solver.transition import solve_point_mdp


def get_health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="imdp-phm-api",
        version="0.4.0",
        modules_ready=["core", "multiscale", "inference", "solver", "benchmark", "empirical", "api"],
    )


def get_defaults() -> DefaultsResponse:
    p = HawkesMertonParams()
    c = StructuralConstants()
    cm = MaintenanceCostModel()
    return DefaultsResponse(
        params=ParamsSchema.from_domain(p),
        constants=ConstantsSchema.from_domain(c),
        cost_model=CostModelSchema.from_domain(cm),
        nasa_ims_test2_se=[float(x) for x in HawkesMertonParams.nasa_ims_test2_se()],
    )


def run_simulation(req: SimulationRequest) -> SimulationResponse:
    params = req.params.to_domain()
    consts = req.constants.to_domain()
    sde = CoupledHawkesMertonSDE(params=params, constants=consts)
    sim = sde.simulate_macro_epochs(n_paths=req.n_paths, t_max=req.t_max, seed=req.seed)

    n_steps = len(sim.time_grid)
    stride = max(1, n_steps // 250)
    t_sub = sim.time_grid[::stride]

    previews: list[TrajectoryPreview] = []
    n_prev = min(req.max_preview_paths, req.n_paths)
    for i in range(n_prev):
        jump_idx = np.flatnonzero(sim.jump_counts[i] > 0).tolist()
        tm = None if np.isnan(sim.tau_major[i]) else round(float(sim.tau_major[i]), 3)
        tc = None if np.isnan(sim.tau_crit[i]) else round(float(sim.tau_crit[i]), 3)
        lih = None if np.isnan(sim.lead_to_imminent_hazard[i]) else round(float(sim.lead_to_imminent_hazard[i]), 3)
        previews.append(
            TrajectoryPreview(
                path_id=i,
                h_series=np.round(sim.h_paths[i, ::stride], 5).tolist(),
                lambda_series=np.round(sim.lambda_paths[i, ::stride], 5).tolist(),
                jump_epochs=jump_idx,
                tau_major=tm,
                tau_crit=tc,
                lead_time_hours=lih,
            )
        )

    valid_crit = sim.tau_crit[~np.isnan(sim.tau_crit)]
    valid_lih = sim.lead_to_imminent_hazard[~np.isnan(sim.lead_to_imminent_hazard)]

    return SimulationResponse(
        time_grid=np.round(t_sub, 4).tolist(),
        trajectories=previews,
        mean_tau_crit_hours=round(float(np.mean(valid_crit)), 3) if len(valid_crit) > 0 else None,
        mean_lead_time_hours=round(float(np.mean(valid_lih)), 3) if len(valid_lih) > 0 else None,
        nominal_branching_ratio=round(params.nominal_branching_ratio(), 4),
        max_branching_ratio=round(params.max_branching_ratio(), 4),
        expected_jump_size=round(params.expected_jump_size(), 5),
    )


def _build_inference_response(
    dataset: MacroEpochDataset,
    consts: StructuralConstants,
    z_mult: float,
    sig_level: float,
) -> tuple[InferenceResponse, CredalParameterSet, object]:
    mle_res = fit_hawkes_merton_mle(dataset, constants=consts, significance_level=sig_level)
    credal = CredalParameterSet.from_mle_result(
        mle_res,
        n_trajectories=dataset.h_paths.shape[0],
        z_mult=z_mult,
    )

    theta_eff = mle_res.effective_params.to_vector()
    intervals = [
        CredalIntervalItem(
            name=PARAM_NAMES[i],
            point_estimate=float(theta_eff[i]),
            standard_error=float(mle_res.standard_errors[i]),
            lower=float(credal.lower[i]),
            upper=float(credal.upper[i]),
        )
        for i in range(8)
    ]

    pb = credal.jump_pbox()
    y_grid = np.linspace(0.005, 0.60, 60, dtype=np.float64)
    ln_y = np.log(y_grid)
    cdf_low = np.minimum(norm.cdf((ln_y - pb.mu_high) / pb.sigma_low), norm.cdf((ln_y - pb.mu_high) / pb.sigma_high))
    cdf_high = np.maximum(norm.cdf((ln_y - pb.mu_low) / pb.sigma_low), norm.cdf((ln_y - pb.mu_low) / pb.sigma_high))

    ht = mle_res.hypothesis_test
    resp = InferenceResponse(
        estimated_params_h1=ParamsSchema.from_domain(mle_res.params_h1),
        effective_params=ParamsSchema.from_domain(mle_res.effective_params),
        hypothesis_test=HypothesisSummary(
            loglik_h0=round(ht.loglik_h0, 3),
            loglik_h1=round(ht.loglik_h1, 3),
            loglik_gain=round(ht.loglik_gain, 3),
            lrt_statistic=round(ht.lrt_statistic, 3),
            p_value=float(ht.p_value),
            delta_bic=round(ht.delta_bic, 3),
            n_active_epochs=ht.n_active_epochs,
            reject_h0=ht.reject_h0,
            active_regime=ht.active_regime,
        ),
        credal_intervals=intervals,
        jump_pbox=JumpPBoxSummary(
            mu_low=pb.mu_low,
            mu_high=pb.mu_high,
            sigma_low=pb.sigma_low,
            sigma_high=pb.sigma_high,
            y_grid=np.round(y_grid, 4).tolist(),
            cdf_lower=np.round(cdf_low, 4).tolist(),
            cdf_upper=np.round(cdf_high, 4).tolist(),
        ),
        total_loglik=round(mle_res.total_loglik, 3),
    )
    return resp, credal, mle_res


def run_inference(req: InferenceRequest) -> InferenceResponse:
    params = req.true_params.to_domain()
    consts = req.constants.to_domain()
    sim = CoupledHawkesMertonSDE(params=params, constants=consts).simulate_macro_epochs(
        n_paths=req.n_trajectories, t_max=req.t_max, seed=req.seed
    )
    ds = MacroEpochDataset.from_simulation(sim, constants=consts)
    resp, _, _ = _build_inference_response(ds, consts, req.z_mult, req.significance_level)
    return resp


def _solve_from_credal(
    credal: CredalParameterSet,
    nominal_params: HawkesMertonParams,
    consts: StructuralConstants,
    cm: MaintenanceCostModel,
    n_h: int,
    n_lam: int,
    lam_min: float,
    lam_max: float,
    tol: float,
    seed: int,
):
    grid = StateGrid2D.uniform(n_h=n_h, n_lam=n_lam, lam_min=lam_min, lam_max=lam_max, constants=consts)
    imdp_res = solve_imdp_minimax(grid=grid, credal_set=credal, cost_model=cm, constants=consts, tol=tol, seed=seed)
    mle_res = solve_point_mdp(grid=grid, params=nominal_params, cost_model=cm, constants=consts, tol=min(tol, 1e-5))

    v_imdp_0 = imdp_res.value_at(grid, consts.h0, nominal_params.mu0)
    v_mle_0 = mle_res.value_at(grid, consts.h0, nominal_params.mu0)

    resp = SolverResponse(
        h_grid=np.round(grid.h_grid, 4).tolist(),
        lam_grid=np.round(grid.lam_grid, 4).tolist(),
        imdp_policy_matrix=imdp_res.optimal_policy.astype(int).tolist(),
        mle_policy_matrix=mle_res.optimal_policy.astype(int).tolist(),
        imdp_value_matrix=np.round(imdp_res.value_function, 2).tolist(),
        mle_value_matrix=np.round(mle_res.value_function, 2).tolist(),
        policy_region_fractions=imdp_res.policy_region_fractions(),
        v_imdp_pristine=round(v_imdp_0, 2),
        v_mle_pristine=round(v_mle_0, 2),
        epistemic_robustness_margin=round(v_imdp_0 - v_mle_0, 2),
        outer_iterations=imdp_res.outer_iterations,
        bellman_residual=float(imdp_res.bellman_residual),
        gp_r2_score=round(imdp_res.gp_r2_score, 5),
        converged=imdp_res.converged,
        solve_time_sec=round(imdp_res.solve_time_sec, 4),
    )
    return resp, grid, imdp_res, mle_res


def run_solver(req: SolverRequest) -> SolverResponse:
    params = req.params.to_domain()
    consts = req.constants.to_domain()
    cm = req.cost_model.to_domain()
    base = params.to_vector()
    if req.standard_errors is not None and len(req.standard_errors) == 8:
        se = np.asarray(req.standard_errors, dtype=np.float64)
    else:
        se = np.array([0.20 * base[0], 0.20 * base[1], 0.18 * base[2], 0.20 * base[3], 0.12 * base[4], 0.20 * base[5], 0.22, 0.12], dtype=np.float64)

    floors = np.array([1e-6, 1e-5, 1e-3, 0.0, 0.05, 0.0, -np.inf, 0.05], dtype=np.float64)
    credal = CredalParameterSet(
        lower=np.maximum(base - req.z_mult * se, floors),
        upper=np.maximum(base + req.z_mult * se, np.maximum(base - req.z_mult * se, floors)),
        max_branching_cap=0.99,
    )
    resp, _, _, _ = _solve_from_credal(
        credal=credal,
        nominal_params=params,
        consts=consts,
        cm=cm,
        n_h=req.n_h,
        n_lam=req.n_lam,
        lam_min=req.lam_min,
        lam_max=req.lam_max,
        tol=req.tol,
        seed=req.seed,
    )
    return resp


def _format_benchmark_response(suites: dict) -> BenchmarkResponse:
    nom_rows = [PolicyBenchmarkRow(**r) for r in suites["Nominal"].summary_table()]
    ls_rows = [PolicyBenchmarkRow(**r) for r in suites["Load-Shift"].summary_table()]
    ls_map = suites["Load-Shift"].metrics_by_policy
    imdp_cvar = ls_map["IMDP-PHM"].cvar_95_cost
    lstm_cvar = max(ls_map["LSTM-RUL"].cvar_95_cost, 1e-6)
    thr_cvar = max(ls_map["Threshold"].cvar_95_cost, 1e-6)

    return BenchmarkResponse(
        nominal_results=nom_rows,
        load_shift_results=ls_rows,
        load_shift_params=ParamsSchema.from_domain(suites["Load-Shift"].true_test_params),
        cvar_reduction_vs_lstm_pct=round(100.0 * (lstm_cvar - imdp_cvar) / lstm_cvar, 2),
        cvar_reduction_vs_threshold_pct=round(100.0 * (thr_cvar - imdp_cvar) / thr_cvar, 2),
    )


def run_benchmark(req: BenchmarkRequest) -> BenchmarkResponse:
    params = req.train_params.to_domain()
    consts = req.constants.to_domain()
    cm = req.cost_model.to_domain()
    sim_train = CoupledHawkesMertonSDE(params=params, constants=consts).simulate_macro_epochs(
        n_paths=req.n_train_paths, t_max=req.horizon_hours, seed=req.seed
    )
    ds_train = MacroEpochDataset.from_simulation(sim_train, constants=consts)
    suites = run_wp3_competitive_benchmark(
        train_dataset=ds_train,
        nominal_test_params=params,
        cost_model=cm,
        constants=consts,
        n_eval_episodes=req.n_eval_episodes,
        horizon_hours=req.horizon_hours,
        lstm_epochs=req.lstm_epochs,
        seed=req.seed,
    )
    return _format_benchmark_response(suites)


def run_full_study(req: FullStudyRequest) -> FullStudyResponse:
    t0 = time.perf_counter()
    params = req.true_params.to_domain()
    consts = req.constants.to_domain()
    cm = req.cost_model.to_domain()

    sim_resp = run_simulation(
        SimulationRequest(
            params=req.true_params,
            constants=req.constants,
            n_paths=req.n_train_paths,
            t_max=req.horizon_hours,
            seed=req.seed,
        )
    )

    sim_raw = CoupledHawkesMertonSDE(params=params, constants=consts).simulate_macro_epochs(
        n_paths=req.n_train_paths, t_max=req.horizon_hours, seed=req.seed
    )
    ds = MacroEpochDataset.from_simulation(sim_raw, constants=consts)
    inf_resp, credal, _ = _build_inference_response(ds, consts, z_mult=1.96, sig_level=0.01)

    solv_resp, grid, _, _ = _solve_from_credal(
        credal=credal,
        nominal_params=inf_resp.effective_params.to_domain(),
        consts=consts,
        cm=cm,
        n_h=33,
        n_lam=19,
        lam_min=0.05,
        lam_max=2.50,
        tol=1e-4,
        seed=req.seed,
    )

    suites = run_wp3_competitive_benchmark(
        train_dataset=ds,
        nominal_test_params=params,
        grid=grid,
        cost_model=cm,
        constants=consts,
        n_eval_episodes=req.n_eval_episodes,
        horizon_hours=req.horizon_hours,
        lstm_epochs=15,
        seed=req.seed,
    )
    bench_resp = _format_benchmark_response(suites)

    return FullStudyResponse(
        simulation=sim_resp,
        inference=inf_resp,
        solver=solv_resp,
        benchmark=bench_resp,
        total_runtime_sec=round(float(time.perf_counter() - t0), 3),
    )


# --- Empirical NASA IMS & PRONOSTIA (FEMTO-ST) Service Layer ---

def _try_resolve_dir(explicit: Optional[str], default_name: str) -> Optional[Path]:
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.exists() else None
    try:
        return resolve_desktop_dir(default_name)
    except FileNotFoundError:
        return None


def get_empirical_catalog(
    ims_dir: Optional[str] = None,
    pronostia_dir: Optional[str] = None,
    cache_dir: str = "results/cache",
) -> EmpiricalCatalogResponse:
    ims_root = _try_resolve_dir(ims_dir, "IMS")
    pron_root = _try_resolve_dir(pronostia_dir, "Pronostia")
    c_dir = Path(cache_dir)

    items: list[EmpiricalCatalogItem] = [
        EmpiricalCatalogItem(
            target_id="COHORT_IMS_FAILED",
            dataset_name="NASA_IMS",
            subset_name="All Failed Bearings (M=4)",
            label="NASA IMS — Failed Bearings Cohort (B1_3, B1_4, B2_1, B3_3)",
            condition_label="2000 rpm / 6000 lbs radial load (Run-to-Failure Cohort)",
            is_cohort=True,
            is_failed_bearing=True,
            cached=all((c_dir / f"IMS_{t}_{b}.npz").exists() for t, b in [("1st_test", "B3"), ("1st_test", "B4"), ("2nd_test", "B1"), ("3rd_test", "B3")]),
        ),
        EmpiricalCatalogItem(
            target_id="CONTROL_IMS_HEALTHY",
            dataset_name="NASA_IMS",
            subset_name="Healthy Controls (M=8)",
            label="NASA IMS — Healthy Control Bearings (Negative Control, M=8)",
            condition_label="2000 rpm / 6000 lbs radial load (Unfailed Channels)",
            is_cohort=True,
            is_failed_bearing=False,
            cached=any((c_dir / "IMS_1st_test_B1.npz").exists() for _ in [0]),
        ),
        EmpiricalCatalogItem(
            target_id="COHORT_PRONOSTIA_ALL",
            dataset_name="PRONOSTIA",
            subset_name="All Bearings (M=17)",
            label="PRONOSTIA (FEMTO-ST) — Full Run-to-Failure Cohort (All 3 Operating Regimes)",
            condition_label="1800/1650/1500 rpm | 4000/4200/5000 N",
            is_cohort=True,
            is_failed_bearing=True,
            cached=(c_dir / "PRONOSTIA_Bearing1_1_max_hv.npz").exists(),
        ),
        EmpiricalCatalogItem(
            target_id="COHORT_PRONOSTIA_COND_1",
            dataset_name="PRONOSTIA",
            subset_name="Condition 1 (M=7)",
            label="PRONOSTIA — Operating Condition 1 Cohort (1800 rpm / 4000 N)",
            condition_label="1800 rpm / 4000 N",
            is_cohort=True,
            is_failed_bearing=True,
            cached=(c_dir / "PRONOSTIA_Bearing1_1_max_hv.npz").exists(),
        ),
        EmpiricalCatalogItem(
            target_id="COHORT_PRONOSTIA_COND_2",
            dataset_name="PRONOSTIA",
            subset_name="Condition 2 (M=7)",
            label="PRONOSTIA — Operating Condition 2 Cohort (1650 rpm / 4200 N)",
            condition_label="1650 rpm / 4200 N",
            is_cohort=True,
            is_failed_bearing=True,
            cached=(c_dir / "PRONOSTIA_Bearing2_1_max_hv.npz").exists(),
        ),
        EmpiricalCatalogItem(
            target_id="COHORT_PRONOSTIA_COND_3",
            dataset_name="PRONOSTIA",
            subset_name="Condition 3 (M=3)",
            label="PRONOSTIA — Operating Condition 3 Cohort (1500 rpm / 5000 N)",
            condition_label="1500 rpm / 5000 N",
            is_cohort=True,
            is_failed_bearing=True,
            cached=(c_dir / "PRONOSTIA_Bearing3_1_max_hv.npz").exists(),
        ),
    ]

    for test_name, bearings in IMS_BEARING_CATALOG.items():
        for b_code, (_, is_failed, desc) in bearings.items():
            tid = f"IMS_{test_name}_{b_code}"
            is_cached = (c_dir / f"{tid}.npz").exists()
            if is_cached or (ims_root is not None and (ims_root / test_name).exists()):
                status_tag = "FAILED" if is_failed else "Healthy"
                items.append(
                    EmpiricalCatalogItem(
                        target_id=tid,
                        dataset_name="NASA_IMS",
                        subset_name=test_name,
                        label=f"{tid} — {desc} [{status_tag}]",
                        condition_label=f"2000 rpm / 6000 lbs | {desc}",
                        is_cohort=False,
                        is_failed_bearing=is_failed,
                        cached=is_cached,
                    )
                )

    pron_names: list[str] = []
    if pron_root is not None:
        pron_names = list(discover_pronostia_bearings(pron_root).keys())
    if not pron_names and c_dir.exists():
        pron_names = sorted(
            p.name.replace("PRONOSTIA_", "").replace("_max_hv.npz", "")
            for p in c_dir.glob("PRONOSTIA_Bearing*_max_hv.npz")
        )

    for b_name in pron_names:
        tid = f"PRONOSTIA_{b_name}"
        cond_digit = b_name.replace("Bearing", "").split("_")[0]
        cond_str = PRONOSTIA_CONDITIONS.get(cond_digit, "Accelerated degradation")
        is_cached = (c_dir / f"{tid}_max_hv.npz").exists()
        items.append(
            EmpiricalCatalogItem(
                target_id=tid,
                dataset_name="PRONOSTIA",
                subset_name=f"Condition_{cond_digit}",
                label=f"{tid} — {cond_str}",
                condition_label=cond_str,
                is_cohort=False,
                is_failed_bearing=True,
                cached=is_cached,
            )
        )

    return EmpiricalCatalogResponse(
        ims_dir=str(ims_root) if ims_root else None,
        pronostia_dir=str(pron_root) if pron_root else None,
        cache_dir=str(c_dir),
        items=items,
    )


def _load_target_records(
    target_id: str,
    consts: StructuralConstants,
    z_thresh: float,
    ims_dir: Optional[str],
    pronostia_dir: Optional[str],
    cache_dir: str,
) -> tuple[list[BearingTrajectoryRecord], str, str, str, bool, bool]:
    ims_root = _try_resolve_dir(ims_dir, "IMS") or Path("~/Desktop/IMS").expanduser()
    pron_root = _try_resolve_dir(pronostia_dir, "Pronostia") or Path("~/Desktop/Pronostia").expanduser()
    c_dir = Path(cache_dir)

    if target_id == "COHORT_IMS_FAILED":
        recs = [
            load_ims_bearing(ims_root, t, b, constants=consts, z_thresh=z_thresh, cache_dir=c_dir)
            for t, b in [("1st_test", "B3"), ("1st_test", "B4"), ("2nd_test", "B1"), ("3rd_test", "B3")]
        ]
        return recs, "NASA_IMS", "All Failed Bearings", "2000 rpm / 6000 lbs (Failed Cohort)", True, True

    if target_id == "CONTROL_IMS_HEALTHY":
        healthy_pairs = [
            (t, b)
            for t, bm in IMS_BEARING_CATALOG.items()
            for b, (_, failed, _) in bm.items()
            if not failed
        ]
        recs = [
            load_ims_bearing(ims_root, t, b, constants=consts, z_thresh=z_thresh, cache_dir=c_dir)
            for t, b in healthy_pairs
        ]
        return recs, "NASA_IMS", "Healthy Controls", "2000 rpm / 6000 lbs (Healthy Controls)", True, False

    if target_id.startswith("IMS_"):
        parts = target_id.split("_")
        if len(parts) < 4:
            raise ValueError(f"Invalid IMS target_id format: {target_id}")
        test_name = f"{parts[1]}_{parts[2]}"
        b_code = parts[3]
        rec = load_ims_bearing(ims_root, test_name, b_code, constants=consts, z_thresh=z_thresh, cache_dir=c_dir)
        return [rec], "NASA_IMS", test_name, rec.condition_label, False, rec.is_failed_bearing

    if target_id.startswith("COHORT_PRONOSTIA_"):
        pron_map = discover_pronostia_bearings(pron_root) if pron_root.exists() else {}
        if not pron_map and c_dir.exists():
            for p in sorted(c_dir.glob("PRONOSTIA_Bearing*_max_hv.npz")):
                bn = p.name.replace("PRONOSTIA_", "").replace("_max_hv.npz", "")
                pron_map[bn] = ("Cached", pron_root / bn)
        if not pron_map:
            raise ValueError(f"No PRONOSTIA bearings found in {pron_root} or cache {c_dir}")

        if target_id == "COHORT_PRONOSTIA_ALL":
            selected = sorted(pron_map.items())
            sub_lbl = "All Conditions (M=17)"
            cond_lbl = "1800/1650/1500 rpm | 4000/4200/5000 N"
        else:
            cond_digit = target_id.split("_")[-1]
            selected = [(k, v) for k, v in sorted(pron_map.items()) if k.startswith(f"Bearing{cond_digit}_")]
            sub_lbl = f"Condition_{cond_digit}"
            cond_lbl = PRONOSTIA_CONDITIONS.get(cond_digit, f"Condition {cond_digit}")

        recs = [
            load_pronostia_bearing(bn, bdir, source_split=sp, constants=consts, z_thresh=z_thresh, cache_dir=c_dir)
            for bn, (sp, bdir) in selected
        ]
        return recs, "PRONOSTIA", sub_lbl, cond_lbl, True, True

    if target_id.startswith("PRONOSTIA_"):
        b_name = target_id.replace("PRONOSTIA_", "")
        pron_map = discover_pronostia_bearings(pron_root) if pron_root.exists() else {}
        if b_name in pron_map:
            sp, bdir = pron_map[b_name]
        else:
            sp, bdir = "Cached", pron_root / b_name
        rec = load_pronostia_bearing(b_name, bdir, source_split=sp, constants=consts, z_thresh=z_thresh, cache_dir=c_dir)
        return [rec], "PRONOSTIA", rec.subset_name, rec.condition_label, False, True

    raise ValueError(f"Unknown empirical target_id: {target_id}")


def run_empirical_study(req: EmpiricalRunRequest) -> EmpiricalRunResponse:
    t0 = time.perf_counter()
    consts = req.constants.to_domain()
    consts = StructuralConstants(
        h0=consts.h0,
        h_crit=consts.h_crit,
        zeta_a=consts.zeta_a,
        theta_a=consts.theta_a,
        zeta_sigma=consts.zeta_sigma,
        dt=consts.dt,
        lambda_th=consts.lambda_th,
        kappa_d=consts.kappa_d,
        z_thresh=req.z_thresh,
    )
    cm = req.cost_model.to_domain()

    records, ds_name, sub_name, cond_lbl, is_cohort, is_failed = _load_target_records(
        target_id=req.target_id,
        consts=consts,
        z_thresh=req.z_thresh,
        ims_dir=req.ims_dir,
        pronostia_dir=req.pronostia_dir,
        cache_dir=req.cache_dir,
    )

    if len(records) == 1:
        dataset = MacroEpochDataset.from_features(records[0].features, constants=consts)
    else:
        dataset = build_cohort_dataset([r.features for r in records], constants=consts)

    inf_resp, credal, mle_res = _build_inference_response(
        dataset=dataset,
        consts=consts,
        z_mult=req.z_mult,
        sig_level=req.significance_level,
    )
    diag = evaluate_point_process_diagnostics(dataset, mle_res, constants=consts, seed=req.seed)
    lam_h0_mat, lam_h1_mat = reconstruct_intensity_paths(dataset, mle_res, constants=consts)

    traj_series: list[EmpiricalTrajectorySeries] = []
    for m, rec in enumerate(records):
        feat = rec.features
        k_m = len(feat.h_hat)
        stride = max(1, k_m // 260)
        idx_sub = np.arange(0, k_m, stride)
        jump_idx = np.flatnonzero(feat.shock_events > 0).tolist()
        traj_series.append(
            EmpiricalTrajectorySeries(
                bearing_id=rec.bearing_id,
                condition_label=rec.condition_label,
                is_failed_bearing=rec.is_failed_bearing,
                duration_hours=round(rec.duration_hours, 3),
                n_snapshots=rec.n_snapshots,
                total_jumps=int(np.sum(feat.shock_events)),
                time_grid=np.round(feat.epoch_times[idx_sub], 4).tolist(),
                v_max_series=np.round(feat.v_max[idx_sub], 5).tolist(),
                v_threshold=round(float(rec.noise_profile.v_th), 5),
                h_series=np.round(feat.h_hat[idx_sub], 5).tolist(),
                lambda_h1_series=np.round(lam_h1_mat[m, idx_sub], 5).tolist(),
                lambda_h0_series=np.round(lam_h0_mat[m, idx_sub], 5).tolist(),
                jump_epochs=jump_idx,
            )
        )

    solv_resp: Optional[SolverResponse] = None
    bench_resp: Optional[BenchmarkResponse] = None
    grid = None

    eff_params = inf_resp.effective_params.to_domain()
    dt_ctrl = float(np.clip(0.20 / max(eff_params.beta, 0.1), 0.005, consts.dt))
    dt_ratio = dt_ctrl / consts.dt
    consts_ctrl = StructuralConstants(
        h0=consts.h0,
        h_crit=consts.h_crit,
        zeta_a=consts.zeta_a,
        theta_a=consts.theta_a,
        zeta_sigma=consts.zeta_sigma,
        dt=dt_ctrl,
        lambda_th=consts.lambda_th,
        kappa_d=consts.kappa_d,
        z_thresh=consts.z_thresh,
    )
    cm_ctrl = MaintenanceCostModel(
        c_oper_base=cm.c_oper_base * dt_ratio,
        c_oper_damage_quad=cm.c_oper_damage_quad * dt_ratio,
        c_inspection=cm.c_inspection,
        inspection_hazard_relief=cm.inspection_hazard_relief,
        c_preventive=cm.c_preventive,
        c_failure=cm.c_failure,
        discount_factor=cm.discount_factor,
    )

    if req.run_wp2 or req.run_wp3:
        dyn_lam_min = min(0.05, max(1e-3, 0.2 * float(credal.lower[2])))
        dyn_lam_max = max(2.50, min(150.0, 2.2 * float(credal.upper[2]) + 1.5 * float(credal.upper[3])))
        solv_resp, grid, _, _ = _solve_from_credal(
            credal=credal,
            nominal_params=eff_params,
            consts=consts_ctrl,
            cm=cm_ctrl,
            n_h=33,
            n_lam=23,
            lam_min=dyn_lam_min,
            lam_max=dyn_lam_max,
            tol=1e-4,
            seed=req.seed,
        )

    if req.run_wp3:
        horizon = float(np.clip(1.25 * np.median([r.duration_hours for r in records]), 4.0, 250.0))
        suites = run_wp3_competitive_benchmark(
            train_dataset=dataset,
            nominal_test_params=eff_params,
            grid=grid,
            cost_model=cm_ctrl,
            constants=consts_ctrl,
            n_eval_episodes=req.n_eval_episodes,
            horizon_hours=horizon,
            lstm_epochs=15,
            seed=req.seed,
        )
        bench_resp = _format_benchmark_response(suites)

    diag_summary = EmpiricalDiagnosticsSummary(
        n_trajectories=diag.n_trajectories,
        n_active_epochs=diag.n_active_epochs,
        total_active_hours=round(diag.total_active_hours, 2),
        total_jumps=diag.total_jumps,
        empirical_jump_rate_per_hour=round(diag.empirical_jump_rate_per_hour, 4),
        nominal_branching_ratio=round(diag.nominal_branching_ratio, 4),
        max_branching_ratio=round(diag.max_branching_ratio, 4),
        n_residuals_h0=diag.n_residuals_h0,
        residual_mean_h0=round(diag.residual_mean_h0, 4),
        dispersion_index_h0=round(diag.dispersion_index_h0, 4),
        ks_stat_h0=round(diag.ks_stat_h0, 4),
        ks_pvalue_h0=round(diag.ks_pvalue_h0, 4),
        lb_stat_5_h0=round(diag.lb_stat_5_h0, 4),
        lb_pvalue_5_h0=round(diag.lb_pvalue_5_h0, 4),
        n_residuals_h1=diag.n_residuals_h1,
        residual_mean_h1=round(diag.residual_mean_h1, 4),
        dispersion_index_h1=round(diag.dispersion_index_h1, 4),
        ks_stat_h1=round(diag.ks_stat_h1, 4),
        ks_pvalue_h1=round(diag.ks_pvalue_h1, 4),
        lb_stat_5_h1=round(diag.lb_stat_5_h1, 4),
        lb_pvalue_5_h1=round(diag.lb_pvalue_5_h1, 4),
        lb_stat_10_h1=round(diag.lb_stat_10_h1, 4),
        lb_pvalue_10_h1=round(diag.lb_pvalue_10_h1, 4),
    )

    return EmpiricalRunResponse(
        target_id=req.target_id,
        dataset_name=ds_name,
        subset_name=sub_name,
        condition_label=cond_lbl,
        is_cohort=is_cohort,
        is_failed_bearing=is_failed,
        z_thresh=req.z_thresh,
        trajectories=traj_series,
        inference=inf_resp,
        diagnostics=diag_summary,
        solver=solv_resp,
        benchmark=bench_resp,
        runtime_sec=round(float(time.perf_counter() - t0), 3),
    )
