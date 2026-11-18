import React, { useState } from 'react';

const POLICY_COLORS = {
  'IMDP-PHM': '#10b981',
  'MLE-MDP': '#38bdf8',
  'LSTM-RUL': '#f43f5e',
  Threshold: '#f59e0b',
};

function CVaRComparisonChart({ rows }) {
  if (!rows?.length) return null;
  const maxVal = Math.max(...rows.map((r) => r.cvar_95), 100);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      {rows.map((r) => {
        const pct = Math.min(100, (r.cvar_95 / maxVal) * 100);
        const color = POLICY_COLORS[r.policy] || '#64748b';
        return (
          <div key={r.policy}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: 4 }}>
              <span style={{ fontWeight: 700, color }}>{r.policy}</span>
              <span style={{ fontFamily: 'JetBrains Mono, monospace' }}>
                CVaR₉₅ = {r.cvar_95.toFixed(2)} | Fail = {(r.failure_rate * 100).toFixed(2)}%
              </span>
            </div>
            <div style={{ background: '#0d1320', borderRadius: 6, height: 14, overflow: 'hidden', border: '1px solid #243047' }}>
              <div
                style={{
                  width: `${pct}%`,
                  height: '100%',
                  background: color,
                  transition: 'width 0.3s ease',
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function WP3BenchmarkPanel({ benchmark }) {
  const [regime, setRegime] = useState('Load-Shift');
  if (!benchmark) return null;

  const rows =
    regime === 'Load-Shift' ? benchmark.load_shift_results : benchmark.nominal_results;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.1rem' }}>
      <div className="kpi-row">
        <div className="kpi-card">
          <div className="kpi-label">Tail CVaR₉₅ Reduction vs. LSTM</div>
          <div className="kpi-value" style={{ color: '#34d399' }}>
            -{benchmark.cvar_reduction_vs_lstm_pct.toFixed(1)}%
          </div>
          <div className="kpi-sub">Under severe Load-Shift stress</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Tail CVaR₉₅ Reduction vs. Threshold</div>
          <div className="kpi-value" style={{ color: '#38bdf8' }}>
            -{benchmark.cvar_reduction_vs_threshold_pct.toFixed(1)}%
          </div>
          <div className="kpi-sub">Eliminates false-alarm NDE inspections</div>
        </div>
      </div>

      <div className="panel-card">
        <div className="panel-header">
          <div>
            <div className="panel-title">Closed-Loop Monte Carlo Policy Benchmark ({regime})</div>
            <div className="panel-desc">
              Paired Common Random Numbers evaluation across IMDP-PHM, MLE-MDP, LSTM-RUL, and Threshold
            </div>
          </div>
          <div style={{ display: 'flex', gap: '0.4rem' }}>
            {['Load-Shift', 'Nominal'].map((tab) => (
              <button
                key={tab}
                className={`tab-btn ${regime === tab ? 'active' : ''}`}
                onClick={() => setRegime(tab)}
              >
                {tab}
              </button>
            ))}
          </div>
        </div>

        <div style={{ marginBottom: '1.25rem' }}>
          <CVaRComparisonChart rows={rows} />
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Policy</th>
                <th>Mean Cost</th>
                <th>CVaR₉₅ Tail</th>
                <th>Discounted</th>
                <th>Failure Rate</th>
                <th>NDE Insp / Ep</th>
                <th>Lead Margin Δτ_LIH</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.policy} className={r.policy === 'IMDP-PHM' ? 'highlight-row' : ''}>
                  <td style={{ fontWeight: 700, color: POLICY_COLORS[r.policy] || '#f1f5f9' }}>
                    {r.policy}
                  </td>
                  <td>{r.mean_cost.toFixed(2)}</td>
                  <td style={{ fontWeight: 700 }}>{r.cvar_95.toFixed(2)}</td>
                  <td>{r.discounted_cost.toFixed(2)}</td>
                  <td>
                    <span className={`badge ${r.failure_rate > 0.05 ? 'badge-danger' : 'badge-hawkes'}`}>
                      {(r.failure_rate * 100).toFixed(2)}%
                    </span>
                  </td>
                  <td>{r.mean_inspections.toFixed(2)}</td>
                  <td>{r.lead_time_h.toFixed(2)} h</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
