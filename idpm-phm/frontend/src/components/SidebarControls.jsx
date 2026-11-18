import React from 'react';
import { Play, RotateCcw, Sliders, Cpu } from 'lucide-react';

const PARAM_LABELS = [
  ['a0', 'a₀ (wear drift)'],
  ['sigma0', 'σ₀ (volatility)'],
  ['mu0', 'μ₀ (base hazard)'],
  ['alpha0', 'α₀ (excitation)'],
  ['beta', 'β (relaxation)'],
  ['gamma', 'γ (feedback)'],
  ['mu_J', 'μ_J (jump mean)'],
  ['sigma_J', 'σ_J (jump std)'],
];

const COST_LABELS = [
  ['c_inspection', 'c_insp (NDE)'],
  ['c_preventive', 'c_prev (replace)'],
  ['c_failure', 'c_fail (failure)'],
  ['discount_factor', 'γ_d (discount)'],
];

export default function SidebarControls({
  params,
  setParams,
  costs,
  setCosts,
  studyConfig,
  setStudyConfig,
  loading,
  onRunFullStudy,
  onResetDefaults,
}) {
  const handleParamChange = (key, raw) => {
    const val = parseFloat(raw);
    if (!Number.isNaN(val)) {
      setParams((prev) => ({ ...prev, [key]: val }));
    }
  };

  const handleCostChange = (key, raw) => {
    const val = parseFloat(raw);
    if (!Number.isNaN(val)) {
      setCosts((prev) => ({ ...prev, [key]: val }));
    }
  };

  return (
    <aside className="sidebar">
      <div className="brand-header">
        <div className="brand-title">
          <Cpu size={20} color="#38bdf8" />
          <span>IMDP-PHM Workbench</span>
        </div>
        <p className="brand-subtitle">
          DFG Sachbeihilfe • TU Dortmund CRE • Imprecise Hawkes-Merton CBM
        </p>
      </div>

      <button
        className="btn-primary"
        onClick={onRunFullStudy}
        disabled={loading}
      >
        <Play size={16} />
        {loading ? 'Running WP1 → WP2 → WP3...' : 'Run Full Study Pipeline'}
      </button>

      <div className="control-group">
        <h3>
          <Sliders size={12} style={{ display: 'inline', marginRight: 6 }} />
          Hawkes-Merton Vector θ ∈ ℝ⁸
        </h3>
        <div className="param-grid">
          {PARAM_LABELS.map(([key, label]) => (
            <div className="field" key={key}>
              <label htmlFor={`param-${key}`}>{label}</label>
              <input
                id={`param-${key}`}
                type="number"
                step="any"
                value={params[key]}
                onChange={(e) => handleParamChange(key, e.target.value)}
              />
            </div>
          ))}
        </div>
      </div>

      <div className="control-group">
        <h3>Asymmetric Cost Model</h3>
        <div className="param-grid">
          {COST_LABELS.map(([key, label]) => (
            <div className="field" key={key}>
              <label htmlFor={`cost-${key}`}>{label}</label>
              <input
                id={`cost-${key}`}
                type="number"
                step="any"
                value={costs[key]}
                onChange={(e) => handleCostChange(key, e.target.value)}
              />
            </div>
          ))}
        </div>
      </div>

      <div className="control-group">
        <h3>Experiment & Monte Carlo</h3>
        <div className="param-grid">
          <div className="field">
            <label htmlFor="cfg-paths">Train Paths (M)</label>
            <input
              id="cfg-paths"
              type="number"
              min={3}
              max={80}
              value={studyConfig.n_train_paths}
              onChange={(e) =>
                setStudyConfig((p) => ({
                  ...p,
                  n_train_paths: Math.max(3, parseInt(e.target.value || '12', 10)),
                }))
              }
            />
          </div>
          <div className="field">
            <label htmlFor="cfg-eval">Eval Episodes</label>
            <input
              id="cfg-eval"
              type="number"
              min={20}
              max={300}
              value={studyConfig.n_eval_episodes}
              onChange={(e) =>
                setStudyConfig((p) => ({
                  ...p,
                  n_eval_episodes: Math.max(20, parseInt(e.target.value || '80', 10)),
                }))
              }
            />
          </div>
          <div className="field">
            <label htmlFor="cfg-horizon">Horizon [h]</label>
            <input
              id="cfg-horizon"
              type="number"
              min={30}
              max={400}
              value={studyConfig.horizon_hours}
              onChange={(e) =>
                setStudyConfig((p) => ({
                  ...p,
                  horizon_hours: Math.max(30, parseFloat(e.target.value || '140')),
                }))
              }
            />
          </div>
          <div className="field">
            <label htmlFor="cfg-seed">PRNG Seed</label>
            <input
              id="cfg-seed"
              type="number"
              value={studyConfig.seed}
              onChange={(e) =>
                setStudyConfig((p) => ({
                  ...p,
                  seed: parseInt(e.target.value || '42', 10),
                }))
              }
            />
          </div>
        </div>
      </div>

      <button className="btn-secondary" onClick={onResetDefaults} disabled={loading}>
        <RotateCcw size={13} style={{ display: 'inline', marginRight: 6 }} />
        Reset Calibrated Defaults
      </button>
    </aside>
  );
}
