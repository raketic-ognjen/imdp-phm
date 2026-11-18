"""Stateless Service Layer connecting FastAPI routes to WP1, WP2, and WP3 scientific engines."""

import time
import numpy as np
from scipy.stats import norm
import torch

torch.set_num_threads(1)

from imdp_phm.api.schemas import (
    BenchmarkRequest,
    BenchmarkResponse,
    ConstantsSchema,
    CostModelSchema,
    CredalIntervalItem,
    DefaultsResponse,
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
from imdp_phm.benchmark.baselines import IMDPPolicy, MLEMDPPolicy
from imdp_phm.benchmark.evaluator import run_wp3_competitive_benchmark
from imdp_phm.core.params import HawkesMertonParams, StructuralConstants
from imdp_phm.core.sde import CoupledHawkesMertonSDE
from imdp_phm.inference.credal import PARAM_NAMES, CredalParameterSet
from imdp_phm.inference.mle import MacroEpochDataset, fit_hawkes_merton_mle
from imdp_phm.solver.bellman import solve_imdp_minimax
from imdp_phm.solver.grid import MaintenanceCostModel, StateGrid2D
from imdp_phm.solver.transition import solve_point_mdp


def get_health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="imdp-phm-api",
        version="0.4.0",
        modules_ready=["core", "multiscale", "inference", "solver", "benchmark", "api"],
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


def _build_inference_response(dataset: MacroEpochDataset, consts: StructuralConstants, z_mult: float, sig_level: float) -> tuple[InferenceResponse, CredalParameterSet]:
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
    return resp, credal


def run_inference(req: InferenceRequest) -> InferenceResponse:
    params = req.true_params.to_domain()
    consts = req.constants.to_domain()
    sim = CoupledHawkesMertonSDE(params=params, constants=consts).simulate_macro_epochs(
        n_paths=req.n_trajectories, t_max=req.t_max, seed=req.seed
    )
    ds = MacroEpochDataset.from_simulation(sim, constants=consts)
    resp, _ = _build_inference_response(ds, consts, req.z_mult, req.significance_level)
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
    inf_resp, credal = _build_inference_response(ds, consts, z_mult=1.96, sig_level=0.01)

    solv_resp, grid, imdp_res, mle_res = _solve_from_credal(
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
        precomputed_imdp_policy=IMDPPolicy(grid=grid, imdp_result=imdp_res, credal_set=credal),
        precomputed_mle_policy=MLEMDPPolicy(grid=grid, point_result=mle_res),
    )
    bench_resp = _format_benchmark_response(suites)

    return FullStudyResponse(
        simulation=sim_resp,
        inference=inf_resp,
        solver=solv_resp,
        benchmark=bench_resp,
        total_runtime_sec=round(float(time.perf_counter() - t0), 3),
    )
