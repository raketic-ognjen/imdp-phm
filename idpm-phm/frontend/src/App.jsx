import React, { useEffect, useState } from 'react';
import { Activity, ShieldCheck, Layers, BarChart3, Wifi, WifiOff } from 'lucide-react';
import {
  apiClient,
  INITIAL_PARAMS,
  INITIAL_COSTS,
  buildFallbackStudy,
} from './api/client.js';
import SidebarControls from './components/SidebarControls.jsx';
import WP1StochasticPanel from './components/WP1StochasticPanel.jsx';
import WP2PolicyHeatmap from './components/WP2PolicyHeatmap.jsx';
import WP3BenchmarkPanel from './components/WP3BenchmarkPanel.jsx';

const TABS = [
  { id: 'overview', label: 'Overview (WP1–WP3)', icon: Layers },
  { id: 'wp1', label: 'WP1: Stochastic & Credal', icon: Activity },
  { id: 'wp2', label: 'WP2: Minimax Policy Map', icon: ShieldCheck },
  { id: 'wp3', label: 'WP3: Load-Shift Benchmark', icon: BarChart3 },
];

export default function App() {
  const [params, setParams] = useState(INITIAL_PARAMS);
  const [costs, setCosts] = useState(INITIAL_COSTS);
  const [studyConfig, setStudyConfig] = useState({
    n_train_paths: 12,
    n_eval_episodes: 80,
    horizon_hours: 140,
    seed: 42,
  });

  const [activeTab, setActiveTab] = useState('overview');
  const [backendLive, setBackendLive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  const [study, setStudy] = useState(() => buildFallbackStudy(INITIAL_PARAMS, INITIAL_COSTS));

  useEffect(() => {
    let mounted = true;
    apiClient
      .health()
      .then((res) => {
        if (mounted && res?.status === 'ok') {
          setBackendLive(true);
        }
      })
      .catch(() => {
        if (mounted) setBackendLive(false);
      });
    return () => {
      mounted = false;
    };
  }, []);

  const handleRunFullStudy = async () => {
    setLoading(true);
    setErrorMsg(null);
    try {
      const payload = {
        true_params: params,
        cost_model: costs,
        n_train_paths: studyConfig.n_train_paths,
        n_eval_episodes: studyConfig.n_eval_episodes,
        horizon_hours: studyConfig.horizon_hours,
        seed: studyConfig.seed,
      };
      const result = await apiClient.runFullStudy(payload);
      setStudy(result);
      setBackendLive(true);
    } catch (err) {
      setBackendLive(false);
      setStudy(buildFallbackStudy(params, costs));
      setErrorMsg(
        `FastAPI backend unreachable (${err.message}). Displaying calibrated local preview.`
      );
    } finally {
      setLoading(false);
    }
  };

  const handleResetDefaults = () => {
    setParams(INITIAL_PARAMS);
    setCosts(INITIAL_COSTS);
    setStudyConfig({ n_train_paths: 12, n_eval_episodes: 80, horizon_hours: 140, seed: 42 });
    setErrorMsg(null);
  };

  const ht = study?.inference?.hypothesis_test;
  const lsImdp = study?.benchmark?.load_shift_results?.find((r) => r.policy === 'IMDP-PHM');

  return (
    <div className="app-shell">
      <SidebarControls
        params={params}
        setParams={setParams}
        costs={costs}
        setCosts={setCosts}
        studyConfig={studyConfig}
        setStudyConfig={setStudyConfig}
        loading={loading}
        onRunFullStudy={handleRunFullStudy}
        onResetDefaults={handleResetDefaults}
      />

      <main className="main-content">
        <header className="topbar">
          <div className="tab-bar">
            {TABS.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                className={`tab-btn ${activeTab === id ? 'active' : ''}`}
                onClick={() => setActiveTab(id)}
              >
                <Icon size={15} />
                {label}
              </button>
            ))}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <span style={{ fontSize: '0.76rem', color: '#94a3b8', fontFamily: 'JetBrains Mono, monospace' }}>
              Runtime: {study?.total_runtime_sec ?? '—'}s
            </span>
            <span className={`status-pill ${backendLive ? 'live' : 'offline'}`}>
              {backendLive ? <Wifi size={13} /> : <WifiOff size={13} />}
              {backendLive ? 'FastAPI Live (:8000)' : 'Preview Mode (Start Uvicorn)'}
            </span>
          </div>
        </header>

        {errorMsg && (
          <div
            style={{
              background: 'rgba(245, 158, 11, 0.12)',
              border: '1px solid rgba(245, 158, 11, 0.4)',
              color: '#fbbf24',
              padding: '0.65rem 1rem',
              borderRadius: 8,
              fontSize: '0.8rem',
            }}
          >
            {errorMsg}
          </div>
        )}

        <section className="kpi-row">
          <div className="kpi-card">
            <div className="kpi-label">Active Stochastic Regime</div>
            <div className="kpi-value" style={{ color: '#34d399' }}>
              {ht?.active_regime || 'NH-Hawkes'}
            </div>
            <div className="kpi-sub">
              LRT = {ht?.lrt_statistic?.toFixed(1)} | ΔBIC = +{ht?.delta_bic?.toFixed(1)}
            </div>
          </div>

          <div className="kpi-card">
            <div className="kpi-label">Minimax Value V*(h₀, μ₀)</div>
            <div className="kpi-value" style={{ color: '#38bdf8' }}>
              {study?.solver?.v_imdp_pristine?.toFixed(2)}
            </div>
            <div className="kpi-sub">
              GP R² = {study?.solver?.gp_r2_score} • {study?.solver?.outer_iterations} iter
            </div>
          </div>

          <div className="kpi-card">
            <div className="kpi-label">Load-Shift Failure Rate</div>
            <div className="kpi-value" style={{ color: '#34d399' }}>
              {((lsImdp?.failure_rate ?? 0) * 100).toFixed(2)}%
            </div>
            <div className="kpi-sub">
              Lead margin Δτ_LIH = {lsImdp?.lead_time_h?.toFixed(2)} h
            </div>
          </div>

          <div className="kpi-card">
            <div className="kpi-label">CVaR₉₅ Tail Savings vs. LSTM</div>
            <div className="kpi-value" style={{ color: '#a78bfa' }}>
              -{study?.benchmark?.cvar_reduction_vs_lstm_pct?.toFixed(1)}%
            </div>
            <div className="kpi-sub">
              -{study?.benchmark?.cvar_reduction_vs_threshold_pct?.toFixed(1)}% vs. Threshold
            </div>
          </div>
        </section>

        {activeTab === 'overview' && (
          <>
            <WP2PolicyHeatmap solver={study?.solver} />
            <WP3BenchmarkPanel benchmark={study?.benchmark} />
            <WP1StochasticPanel simulation={study?.simulation} inference={study?.inference} />
          </>
        )}

        {activeTab === 'wp1' && (
          <WP1StochasticPanel simulation={study?.simulation} inference={study?.inference} />
        )}

        {activeTab === 'wp2' && <WP2PolicyHeatmap solver={study?.solver} />}

        {activeTab === 'wp3' && <WP3BenchmarkPanel benchmark={study?.benchmark} />}
      </main>
    </div>
  );
}
