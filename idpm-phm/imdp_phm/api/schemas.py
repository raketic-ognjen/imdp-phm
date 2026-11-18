"""Pydantic v2 Request & Response Schemas for IMDP-PHM FastAPI Backend."""

from typing import Optional
from pydantic import BaseModel, Field

from imdp_phm.core.params import HawkesMertonParams, StructuralConstants
from imdp_phm.solver.grid import MaintenanceCostModel


class ParamsSchema(BaseModel):
    a0: float = Field(default=3.12e-4, gt=0.0, description="Baseline continuous wear drift [h^-1]")
    sigma0: float = Field(default=1.18e-3, gt=0.0, description="Baseline Brownian volatility [h^-1/2]")
    mu0: float = Field(default=0.15, gt=0.0, description="Nominal exogenous hazard rate [h^-1]")
    alpha0: float = Field(default=0.35, ge=0.0, description="Baseline shock excitation sensitivity [h^-1]")
    beta: float = Field(default=0.85, gt=0.0, description="Elastodynamic stress-relaxation rate [h^-1]")
    gamma: float = Field(default=1.40, ge=0.0, description="Quadratic structural feedback gain [-]")
    mu_J: float = Field(default=-3.344, description="Log-normal jump severity mean [-]")
    sigma_J: float = Field(default=0.996, gt=0.0, description="Log-normal jump severity std [-]")

    def to_domain(self) -> HawkesMertonParams:
        return HawkesMertonParams(
            a0=self.a0,
            sigma0=self.sigma0,
            mu0=self.mu0,
            alpha0=self.alpha0,
            beta=self.beta,
            gamma=self.gamma,
            mu_J=self.mu_J,
            sigma_J=self.sigma_J,
        )

    @classmethod
    def from_domain(cls, p: HawkesMertonParams) -> "ParamsSchema":
        return cls(
            a0=p.a0,
            sigma0=p.sigma0,
            mu0=p.mu0,
            alpha0=p.alpha0,
            beta=p.beta,
            gamma=p.gamma,
            mu_J=p.mu_J,
            sigma_J=p.sigma_J,
        )


class ConstantsSchema(BaseModel):
    h0: float = Field(default=0.10, ge=0.0)
    h_crit: float = Field(default=2.50, gt=0.1)
    zeta_a: float = Field(default=1.85, ge=0.0)
    theta_a: float = Field(default=1.50, gt=0.0)
    zeta_sigma: float = Field(default=2.10, ge=0.0)
    dt: float = Field(default=0.1667, gt=1e-4)
    lambda_th: float = Field(default=0.40, gt=0.0)
    kappa_d: float = Field(default=0.042, gt=0.0)
    z_thresh: float = Field(default=4.5, gt=0.0)

    def to_domain(self) -> StructuralConstants:
        return StructuralConstants(
            h0=self.h0,
            h_crit=self.h_crit,
            zeta_a=self.zeta_a,
            theta_a=self.theta_a,
            zeta_sigma=self.zeta_sigma,
            dt=self.dt,
            lambda_th=self.lambda_th,
            kappa_d=self.kappa_d,
            z_thresh=self.z_thresh,
        )

    @classmethod
    def from_domain(cls, c: StructuralConstants) -> "ConstantsSchema":
        return cls(
            h0=c.h0,
            h_crit=c.h_crit,
            zeta_a=c.zeta_a,
            theta_a=c.theta_a,
            zeta_sigma=c.zeta_sigma,
            dt=c.dt,
            lambda_th=c.lambda_th,
            kappa_d=c.kappa_d,
            z_thresh=c.z_thresh,
        )


class CostModelSchema(BaseModel):
    c_oper_base: float = Field(default=1.0, ge=0.0)
    c_oper_damage_quad: float = Field(default=8.0, ge=0.0)
    c_inspection: float = Field(default=25.0, ge=0.0)
    inspection_hazard_relief: float = Field(default=0.55, ge=0.0, le=1.0)
    c_preventive: float = Field(default=250.0, gt=0.0)
    c_failure: float = Field(default=3500.0, gt=0.0)
    discount_factor: float = Field(default=0.995, gt=0.5, lt=1.0)

    def to_domain(self) -> MaintenanceCostModel:
        return MaintenanceCostModel(
            c_oper_base=self.c_oper_base,
            c_oper_damage_quad=self.c_oper_damage_quad,
            c_inspection=self.c_inspection,
            inspection_hazard_relief=self.inspection_hazard_relief,
            c_preventive=self.c_preventive,
            c_failure=self.c_failure,
            discount_factor=self.discount_factor,
        )

    @classmethod
    def from_domain(cls, cm: MaintenanceCostModel) -> "CostModelSchema":
        return cls(
            c_oper_base=cm.c_oper_base,
            c_oper_damage_quad=cm.c_oper_damage_quad,
            c_inspection=cm.c_inspection,
            inspection_hazard_relief=cm.inspection_hazard_relief,
            c_preventive=cm.c_preventive,
            c_failure=cm.c_failure,
            discount_factor=cm.discount_factor,
        )


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    modules_ready: list[str]


class DefaultsResponse(BaseModel):
    params: ParamsSchema
    constants: ConstantsSchema
    cost_model: CostModelSchema
    nasa_ims_test2_se: list[float]


class SimulationRequest(BaseModel):
    params: ParamsSchema = Field(default_factory=ParamsSchema)
    constants: ConstantsSchema = Field(default_factory=ConstantsSchema)
    n_paths: int = Field(default=16, ge=2, le=500)
    t_max: float = Field(default=150.0, gt=5.0, le=1000.0)
    max_preview_paths: int = Field(default=6, ge=1, le=20)
    seed: int = Field(default=42)


class TrajectoryPreview(BaseModel):
    path_id: int
    h_series: list[float]
    lambda_series: list[float]
    jump_epochs: list[int]
    tau_major: Optional[float]
    tau_crit: Optional[float]
    lead_time_hours: Optional[float]


class SimulationResponse(BaseModel):
    time_grid: list[float]
    trajectories: list[TrajectoryPreview]
    mean_tau_crit_hours: Optional[float]
    mean_lead_time_hours: Optional[float]
    nominal_branching_ratio: float
    max_branching_ratio: float
    expected_jump_size: float


class InferenceRequest(BaseModel):
    true_params: ParamsSchema = Field(
        default_factory=lambda: ParamsSchema(
            a0=2.8e-3, sigma0=2.0e-3, mu0=0.30, alpha0=0.42, beta=0.90, gamma=1.35, mu_J=-2.3, sigma_J=0.58
        )
    )
    constants: ConstantsSchema = Field(default_factory=ConstantsSchema)
    n_trajectories: int = Field(default=12, ge=2, le=200)
    t_max: float = Field(default=150.0, gt=10.0, le=600.0)
    z_mult: float = Field(default=1.96, gt=0.5, le=4.0)
    significance_level: float = Field(default=0.01, gt=1e-6, lt=0.25)
    seed: int = Field(default=55)


class HypothesisSummary(BaseModel):
    loglik_h0: float
    loglik_h1: float
    loglik_gain: float
    lrt_statistic: float
    p_value: float
    delta_bic: float
    n_active_epochs: int
    reject_h0: bool
    active_regime: str


class CredalIntervalItem(BaseModel):
    name: str
    point_estimate: float
    standard_error: float
    lower: float
    upper: float


class JumpPBoxSummary(BaseModel):
    mu_low: float
    mu_high: float
    sigma_low: float
    sigma_high: float
    y_grid: list[float]
    cdf_lower: list[float]
    cdf_upper: list[float]


class InferenceResponse(BaseModel):
    estimated_params_h1: ParamsSchema
    effective_params: ParamsSchema
    hypothesis_test: HypothesisSummary
    credal_intervals: list[CredalIntervalItem]
    jump_pbox: JumpPBoxSummary
    total_loglik: float


class SolverRequest(BaseModel):
    params: ParamsSchema = Field(
        default_factory=lambda: ParamsSchema(
            a0=2.5e-3, sigma0=2.0e-3, mu0=0.25, alpha0=0.40, beta=0.90, gamma=1.30, mu_J=-2.3, sigma_J=0.60
        )
    )
    standard_errors: Optional[list[float]] = Field(
        default=None,
        description="8-element SE vector; defaults to calibrated epistemic uncertainty bounds.",
    )
    constants: ConstantsSchema = Field(default_factory=ConstantsSchema)
    cost_model: CostModelSchema = Field(default_factory=CostModelSchema)
    n_h: int = Field(default=35, ge=11, le=81)
    n_lam: int = Field(default=21, ge=9, le=51)
    lam_min: float = Field(default=0.05, gt=0.0)
    lam_max: float = Field(default=2.50, gt=0.5)
    z_mult: float = Field(default=1.96, gt=0.1, le=4.0)
    tol: float = Field(default=1e-5, gt=1e-9, lt=1e-1)
    seed: int = Field(default=42)


class SolverResponse(BaseModel):
    h_grid: list[float]
    lam_grid: list[float]
    imdp_policy_matrix: list[list[int]]
    mle_policy_matrix: list[list[int]]
    imdp_value_matrix: list[list[float]]
    mle_value_matrix: list[list[float]]
    policy_region_fractions: dict[str, float]
    v_imdp_pristine: float
    v_mle_pristine: float
    epistemic_robustness_margin: float
    outer_iterations: int
    bellman_residual: float
    gp_r2_score: float
    converged: bool
    solve_time_sec: float


class BenchmarkRequest(BaseModel):
    train_params: ParamsSchema = Field(
        default_factory=lambda: ParamsSchema(
            a0=2.8e-3, sigma0=2.0e-3, mu0=0.30, alpha0=0.40, beta=0.90, gamma=1.30, mu_J=-2.3, sigma_J=0.58
        )
    )
    constants: ConstantsSchema = Field(default_factory=ConstantsSchema)
    cost_model: CostModelSchema = Field(default_factory=CostModelSchema)
    n_train_paths: int = Field(default=12, ge=3, le=100)
    n_eval_episodes: int = Field(default=100, ge=20, le=500)
    horizon_hours: float = Field(default=140.0, gt=20.0, le=500.0)
    lstm_epochs: int = Field(default=20, ge=5, le=100)
    seed: int = Field(default=42)


class PolicyBenchmarkRow(BaseModel):
    policy: str
    regime: str
    mean_cost: float
    std_cost: float
    cvar_95: float
    discounted_cost: float
    failure_rate: float
    premature_rep_rate: float
    mean_inspections: float
    mean_replacements: float
    service_life_h: float
    lead_time_h: float


class BenchmarkResponse(BaseModel):
    nominal_results: list[PolicyBenchmarkRow]
    load_shift_results: list[PolicyBenchmarkRow]
    load_shift_params: ParamsSchema
    cvar_reduction_vs_lstm_pct: float
    cvar_reduction_vs_threshold_pct: float


class FullStudyRequest(BaseModel):
    true_params: ParamsSchema = Field(
        default_factory=lambda: ParamsSchema(
            a0=2.8e-3, sigma0=2.0e-3, mu0=0.30, alpha0=0.40, beta=0.90, gamma=1.30, mu_J=-2.3, sigma_J=0.58
        )
    )
    constants: ConstantsSchema = Field(default_factory=ConstantsSchema)
    cost_model: CostModelSchema = Field(default_factory=CostModelSchema)
    n_train_paths: int = Field(default=12, ge=3, le=80)
    n_eval_episodes: int = Field(default=80, ge=20, le=300)
    horizon_hours: float = Field(default=140.0, gt=20.0, le=400.0)
    seed: int = Field(default=42)


class FullStudyResponse(BaseModel):
    simulation: SimulationResponse
    inference: InferenceResponse
    solver: SolverResponse
    benchmark: BenchmarkResponse
    total_runtime_sec: float
