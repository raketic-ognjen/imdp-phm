import React from 'react';

const PATH_COLORS = ['#38bdf8', '#10b981', '#f59e0b', '#a78bfa', '#fb7185'];

function TrajectoryChart({ timeGrid, trajectories, hCrit = 2.5 }) {
  if (!timeGrid?.length || !trajectories?.length) return null;
  const W = 540;
  const H = 220;
  const pad = { top: 16, right: 20, bottom: 32, left: 44 };
  const innerW = W - pad.left - pad.right;
  const innerH = H - pad.top - pad.bottom;

  const tMax = Math.max(...timeGrid, 1);
  const yMax = Math.max(hCrit * 1.08, 2.7);
  const toX = (t) => pad.left + (t / tMax) * innerW;
  const toY = (v) => pad.top + innerH - (Math.min(v, yMax) / yMax) * innerH;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      {[0, 0.5, 1.0, 1.5, 2.0, 2.5].map((tick) => (
        <g key={tick}>
          <line
            x1={pad.left}
            y1={toY(tick)}
            x2={W - pad.right}
            y2={toY(tick)}
            stroke="#1e293b"
            strokeDasharray={tick === hCrit ? '4 3' : undefined}
          />
          <text x={pad.left - 8} y={toY(tick) + 4} textAnchor="end" fontSize="10" fill="#94a3b8">
            {tick.toFixed(1)}
          </text>
        </g>
      ))}
      <line
        x1={pad.left}
        y1={toY(hCrit)}
        x2={W - pad.right}
        y2={toY(hCrit)}
        stroke="#f43f5e"
        strokeWidth="1.5"
        strokeDasharray="5 4"
      />
      <text x={W - pad.right - 4} y={toY(hCrit) - 5} textAnchor="end" fontSize="10" fill="#fb7185">
        H_crit = {hCrit.toFixed(2)}
      </text>

      {trajectories.map((tr, idx) => {
        const pts = tr.h_series
          .map((hVal, k) => `${toX(timeGrid[k] ?? 0).toFixed(1)},${toY(hVal).toFixed(1)}`)
          .join(' ');
        return (
          <polyline
            key={tr.path_id}
            fill="none"
            stroke={PATH_COLORS[idx % PATH_COLORS.length]}
            strokeWidth="2"
            points={pts}
          />
        );
      })}

      <text x={W / 2} y={H - 6} textAnchor="middle" fontSize="11" fill="#94a3b8">
        Operational Time t [hours]
      </text>
    </svg>
  );
}

function PBoxChart({ pbox }) {
  if (!pbox?.y_grid?.length) return null;
  const W = 520;
  const H = 210;
  const pad = { top: 16, right: 20, bottom: 32, left: 44 };
  const innerW = W - pad.left - pad.right;
  const innerH = H - pad.top - pad.bottom;

  const yMax = Math.max(...pbox.y_grid, 0.1);
  const toX = (y) => pad.left + (y / yMax) * innerW;
  const toY = (cdf) => pad.top + innerH - cdf * innerH;

  const upperPts = pbox.y_grid.map((y, i) => `${toX(y).toFixed(1)},${toY(pbox.cdf_upper[i]).toFixed(1)}`);
  const lowerPtsRev = [...pbox.y_grid]
    .reverse()
    .map((y, revI) => {
      const origI = pbox.y_grid.length - 1 - revI;
      return `${toX(y).toFixed(1)},${toY(pbox.cdf_lower[origI]).toFixed(1)}`;
    });

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      {[0, 0.25, 0.5, 0.75, 1.0].map((c) => (
        <g key={c}>
          <line x1={pad.left} y1={toY(c)} x2={W - pad.right} y2={toY(c)} stroke="#1e293b" />
          <text x={pad.left - 8} y={toY(c) + 4} textAnchor="end" fontSize="10" fill="#94a3b8">
            {c.toFixed(2)}
          </text>
        </g>
      ))}
      <polygon
        points={[...upperPts, ...lowerPtsRev].join(' ')}
        fill="rgba(56, 189, 248, 0.18)"
        stroke="none"
      />
      <polyline fill="none" stroke="#38bdf8" strokeWidth="2" points={upperPts.join(' ')} />
      <polyline
        fill="none"
        stroke="#f59e0b"
        strokeWidth="2"
        points={pbox.y_grid.map((y, i) => `${toX(y).toFixed(1)},${toY(pbox.cdf_lower[i]).toFixed(1)}`).join(' ')}
      />
      <text x={W / 2} y={H - 6} textAnchor="middle" fontSize="11" fill="#94a3b8">
        Jump Severity y (Parametric P-Box [F̲_J(y), F̄_J(y)])
      </text>
    </svg>
  );
}

export default function WP1StochasticPanel({ simulation, inference }) {
  const ht = inference?.hypothesis_test;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.1rem' }}>
      <div className="two-col">
        <div className="panel-card">
          <div className="panel-header">
            <div>
              <div className="panel-title">Coupled Hawkes-Merton Trajectories H_t</div>
              <div className="panel-desc">
                Macro-epoch cumulative damage paths with self-exciting jump cascades
              </div>
            </div>
            <span className="badge badge-hawkes">
              n_max = {simulation?.max_branching_ratio ?? '—'}
            </span>
          </div>
          <TrajectoryChart
            timeGrid={simulation?.time_grid}
            trajectories={simulation?.trajectories}
          />
        </div>

        <div className="panel-card">
          <div className="panel-header">
            <div>
              <div className="panel-title">WP1 Hypothesis Test & Jump P-Box F_J</div>
              <div className="panel-desc">
                H₀: α₀ = 0 (NHPP) vs. H₁: α₀ &gt; 0 (NH-Hawkes) • LRT &amp; ΔBIC Selection
              </div>
            </div>
            <span className={`badge ${ht?.reject_h0 ? 'badge-hawkes' : 'badge-nhpp'}`}>
              {ht?.active_regime || 'NH-Hawkes'}
            </span>
          </div>

          {ht && (
            <div className="kpi-row" style={{ marginBottom: '0.85rem' }}>
              <div className="kpi-card">
                <div className="kpi-label">LRT Statistic</div>
                <div className="kpi-value">{ht.lrt_statistic.toFixed(2)}</div>
                <div className="kpi-sub">p = {ht.p_value.toExponential(2)}</div>
              </div>
              <div className="kpi-card">
                <div className="kpi-label">ΔBIC Gain</div>
                <div className="kpi-value" style={{ color: '#34d399' }}>
                  +{ht.delta_bic.toFixed(2)}
                </div>
                <div className="kpi-sub">{ht.n_active_epochs} active epochs</div>
              </div>
            </div>
          )}

          <PBoxChart pbox={inference?.jump_pbox} />
        </div>
      </div>

      <div className="panel-card">
        <div className="panel-header">
          <div>
            <div className="panel-title">Epistemic Credal Parameter Set Θ ⊂ ℝ₊⁸</div>
            <div className="panel-desc">
              Small-sample inflated confidence intervals around effective MLE θ̂
            </div>
          </div>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Parameter</th>
                <th>Point MLE (θ̂)</th>
                <th>Fisher SE</th>
                <th>Credal Lower (θ̲)</th>
                <th>Credal Upper (θ̄)</th>
              </tr>
            </thead>
            <tbody>
              {(inference?.credal_intervals || []).map((item) => (
                <tr key={item.name}>
                  <td style={{ fontWeight: 700, color: '#38bdf8' }}>{item.name}</td>
                  <td>{item.point_estimate.toExponential(3)}</td>
                  <td>{item.standard_error.toExponential(3)}</td>
                  <td>{item.lower.toExponential(3)}</td>
                  <td>{item.upper.toExponential(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
