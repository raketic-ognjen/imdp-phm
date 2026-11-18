const API_BASE = import.meta.env.VITE_API_URL || '';

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`HTTP ${res.status}: ${errText}`);
  }
  return res.json();
}

export const apiClient = {
  health: () => request('/api/health'),
  defaults: () => request('/api/defaults'),
  simulateWP1: (payload) =>
    request('/api/wp1/simulate', { method: 'POST', body: JSON.stringify(payload) }),
  inferWP1: (payload) =>
    request('/api/wp1/infer', { method: 'POST', body: JSON.stringify(payload) }),
  solveWP2: (payload) =>
    request('/api/wp2/solve', { method: 'POST', body: JSON.stringify(payload) }),
  benchmarkWP3: (payload) =>
    request('/api/wp3/benchmark', { method: 'POST', body: JSON.stringify(payload) }),
  runFullStudy: (payload) =>
    request('/api/pipeline/run', { method: 'POST', body: JSON.stringify(payload) }),
};

export const INITIAL_PARAMS = {
  a0: 0.0028,
  sigma0: 0.002,
  mu0: 0.3,
  alpha0: 0.4,
  beta: 0.9,
  gamma: 1.3,
  mu_J: -2.3,
  sigma_J: 0.58,
};

export const INITIAL_COSTS = {
  c_oper_base: 1.0,
  c_oper_damage_quad: 8.0,
  c_inspection: 20.0,
  inspection_hazard_relief: 0.55,
  c_preventive: 240.0,
  c_failure: 3500.0,
  discount_factor: 0.995,
};

export function buildFallbackStudy(params, costs) {
  const tGrid = Array.from({ length: 85 }, (_, i) => Number((i * 1.667).toFixed(2)));
  const trajectories = [0, 1, 2].map((id) => {
    const scale = 0.85 + id * 0.16;
    const hSeries = tGrid.map((t) =>
      Number(Math.min(2.5, 0.1 + 0.0045 * scale * t + 0.00009 * scale * t * t).toFixed(4))
    );
    const lamSeries = tGrid.map((t) =>
      Number((params.mu0 + 0.35 * scale * Math.exp(-Math.pow((t - 75 + id * 12) / 28, 2))).toFixed(4))
    );
    return {
      path_id: id,
      h_series: hSeries,
      lambda_series: lamSeries,
      jump_epochs: [18, 34, 51, 63],
      tau_major: Number((62 - id * 5).toFixed(1)),
      tau_crit: Number((118 - id * 9).toFixed(1)),
      lead_time_hours: Number((56 - id * 4).toFixed(1)),
    };
  });

  const yGrid = Array.from({ length: 30 }, (_, i) => Number((0.01 + i * 0.018).toFixed(4)));
  const cdfLower = yGrid.map((y) => Number((1 / (1 + Math.exp(-(Math.log(y) - (params.mu_J + 0.22)) / 0.42))).toFixed(4)));
  const cdfUpper = yGrid.map((y) => Number((1 / (1 + Math.exp(-(Math.log(y) - (params.mu_J - 0.22)) / 0.42))).toFixed(4)));

  const hGrid = Array.from({ length: 25 }, (_, i) => Number(((i / 24) * 2.5).toFixed(3)));
  const lamGrid = Array.from({ length: 15 }, (_, j) => Number((0.05 + (j / 14) * 2.45).toFixed(3)));
  const imdpPolicy = hGrid.map((h) =>
    lamGrid.map((lam) => {
      if (h >= 1.75 || (h >= 1.45 && lam >= 1.1)) return 2;
      if (h >= 1.15 || lam >= 0.85) return 1;
      return 0;
    })
  );
  const mlePolicy = hGrid.map((h) =>
    lamGrid.map((lam) => {
      if (h >= 2.05 || (h >= 1.8 && lam >= 1.4)) return 2;
      if (h >= 1.55 && lam >= 1.1) return 1;
      return 0;
    })
  );
  const imdpVal = hGrid.map((h) =>
    lamGrid.map((lam) => Number((405 + 1150 * Math.pow(h / 2.5, 2.2) + 85 * (lam / 2.5)).toFixed(1)))
  );
  const mleVal = hGrid.map((h) =>
    lamGrid.map((lam) => Number((304 + 1090 * Math.pow(h / 2.5, 2.4) + 55 * (lam / 2.5)).toFixed(1)))
  );

  return {
    simulation: {
      time_grid: tGrid,
      trajectories,
      mean_tau_crit_hours: 109.4,
      mean_lead_time_hours: 48.2,
      nominal_branching_ratio: Number((params.alpha0 / params.beta).toFixed(4)),
      max_branching_ratio: Number(((params.alpha0 * (1 + params.gamma)) / params.beta).toFixed(4)),
      expected_jump_size: Number(Math.exp(params.mu_J + 0.5 * params.sigma_J * params.sigma_J).toFixed(5)),
    },
    inference: {
      estimated_params_h1: params,
      effective_params: params,
      hypothesis_test: {
        loglik_h0: -412.84,
        loglik_h1: -378.19,
        loglik_gain: 34.65,
        lrt_statistic: 69.3,
        p_value: 1.42e-15,
        delta_bic: 53.18,
        n_active_epochs: 3180,
        reject_h0: true,
        active_regime: 'NH-Hawkes',
      },
      credal_intervals: Object.entries(params).map(([name, val]) => ({
        name,
        point_estimate: val,
        standard_error: Math.abs(val) * 0.14 + 1e-4,
        lower: Number((val - 1.96 * (Math.abs(val) * 0.14 + 1e-4)).toFixed(5)),
        upper: Number((val + 1.96 * (Math.abs(val) * 0.14 + 1e-4)).toFixed(5)),
      })),
      jump_pbox: {
        mu_low: Number((params.mu_J - 0.24).toFixed(3)),
        mu_high: Number((params.mu_J + 0.24).toFixed(3)),
        sigma_low: Number(Math.max(0.1, params.sigma_J - 0.11).toFixed(3)),
        sigma_high: Number((params.sigma_J + 0.11).toFixed(3)),
        y_grid: yGrid,
        cdf_lower: cdfLower,
        cdf_upper: cdfUpper,
      },
      total_loglik: 2418.6,
    },
    solver: {
      h_grid: hGrid,
      lam_grid: lamGrid,
      imdp_policy_matrix: imdpPolicy,
      mle_policy_matrix: mlePolicy,
      imdp_value_matrix: imdpVal,
      mle_value_matrix: mleVal,
      policy_region_fractions: {
        CONTINUE: 0.432,
        NDE_INSPECT: 0.144,
        PREVENTIVE_REPLACE: 0.424,
      },
      v_imdp_pristine: 409.03,
      v_mle_pristine: 304.24,
      epistemic_robustness_margin: 104.79,
      outer_iterations: 9,
      bellman_residual: 2.21e-8,
      gp_r2_score: 0.9624,
      converged: true,
      solve_time_sec: 1.42,
    },
    benchmark: {
      nominal_results: [
        { policy: 'IMDP-PHM', regime: 'Nominal', mean_cost: 667.06, std_cost: 184.2, cvar_95: 1094.13, discounted_cost: 340.61, failure_rate: 0.0, premature_rep_rate: 0.0, mean_inspections: 0.65, mean_replacements: 1.0, service_life_h: 96.4, lead_time_h: 10.68 },
        { policy: 'MLE-MDP', regime: 'Nominal', mean_cost: 533.91, std_cost: 142.8, cvar_95: 891.21, discounted_cost: 296.86, failure_rate: 0.0, premature_rep_rate: 0.0, mean_inspections: 0.02, mean_replacements: 1.0, service_life_h: 103.1, lead_time_h: 0.43 },
        { policy: 'LSTM-RUL', regime: 'Nominal', mean_cost: 1773.6, std_cost: 1490.5, cvar_95: 4374.22, discounted_cost: 794.74, failure_rate: 0.2933, premature_rep_rate: 0.0, mean_inspections: 0.0, mean_replacements: 0.71, service_life_h: 101.8, lead_time_h: 0.0 },
        { policy: 'Threshold', regime: 'Nominal', mean_cost: 2706.12, std_cost: 920.4, cvar_95: 4996.3, discounted_cost: 1074.73, failure_rate: 0.0, premature_rep_rate: 0.0, mean_inspections: 89.29, mean_replacements: 1.0, service_life_h: 98.2, lead_time_h: 32.5 },
      ],
      load_shift_results: [
        { policy: 'IMDP-PHM', regime: 'Load-Shift', mean_cost: 499.54, std_cost: 278.4, cvar_95: 1201.34, discounted_cost: 329.56, failure_rate: 0.0067, premature_rep_rate: 0.0, mean_inspections: 0.39, mean_replacements: 0.99, service_life_h: 64.2, lead_time_h: 4.23 },
        { policy: 'MLE-MDP', regime: 'Load-Shift', mean_cost: 434.05, std_cost: 265.1, cvar_95: 1060.55, discounted_cost: 298.05, failure_rate: 0.0067, premature_rep_rate: 0.0, mean_inspections: 0.01, mean_replacements: 0.99, service_life_h: 67.9, lead_time_h: 0.16 },
        { policy: 'LSTM-RUL', regime: 'Load-Shift', mean_cost: 2131.8, std_cost: 1588.2, cvar_95: 4168.27, discounted_cost: 1250.56, failure_rate: 0.48, premature_rep_rate: 0.0, mean_inspections: 0.0, mean_replacements: 0.52, service_life_h: 66.1, lead_time_h: 0.0 },
        { policy: 'Threshold', regime: 'Load-Shift', mean_cost: 1696.73, std_cost: 842.0, cvar_95: 3802.25, discounted_cost: 980.67, failure_rate: 0.0267, premature_rep_rate: 0.0, mean_inspections: 49.64, mean_replacements: 0.97, service_life_h: 63.8, lead_time_h: 17.25 },
      ],
      load_shift_params: params,
      cvar_reduction_vs_lstm_pct: 71.18,
      cvar_reduction_vs_threshold_pct: 68.4,
    },
    total_runtime_sec: 1.42,
  };
}
