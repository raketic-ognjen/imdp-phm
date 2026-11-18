import React, { useState } from 'react';

const ACTION_META = {
  0: { label: 'CONTINUE', color: '#0ea5e9' },
  1: { label: 'NDE_INSPECT', color: '#f59e0b' },
  2: { label: 'PREVENTIVE_REPLACE', color: '#f43f5e' },
};

function PolicyGridSVG({ title, subtitle, hGrid, lamGrid, policyMatrix, valueMatrix, onHoverCell }) {
  if (!hGrid?.length || !lamGrid?.length || !policyMatrix?.length) return null;

  const W = 480;
  const H = 290;
  const pad = { top: 18, right: 16, bottom: 38, left: 48 };
  const innerW = W - pad.left - pad.right;
  const innerH = H - pad.top - pad.bottom;

  const nH = hGrid.length;
  const nL = lamGrid.length;
  const cellW = innerW / nH;
  const cellH = innerH / nL;

  return (
    <div className="panel-card">
      <div className="panel-header">
        <div>
          <div className="panel-title">{title}</div>
          <div className="panel-desc">{subtitle}</div>
        </div>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        {policyMatrix.map((row, ih) =>
          row.map((act, il) => {
            const x = pad.left + ih * cellW;
            const y = pad.top + (nL - 1 - il) * cellH;
            const fill = ACTION_META[act]?.color || '#334155';
            return (
              <rect
                key={`${ih}-${il}`}
                x={x}
                y={y}
                width={cellW + 0.4}
                height={cellH + 0.4}
                fill={fill}
                fillOpacity={0.86}
                onMouseEnter={() =>
                  onHoverCell?.({
                    h: hGrid[ih],
                    lam: lamGrid[il],
                    action: ACTION_META[act]?.label || String(act),
                    value: valueMatrix?.[ih]?.[il],
                  })
                }
              />
            );
          })
        )}

        {[0, 0.25, 0.5, 0.75, 1.0].map((frac) => {
          const ih = Math.min(Math.round(frac * (nH - 1)), nH - 1);
          const x = pad.left + (ih + 0.5) * cellW;
          return (
            <text key={`h-${frac}`} x={x} y={H - 18} textAnchor="middle" fontSize="10" fill="#94a3b8">
              {hGrid[ih].toFixed(2)}
            </text>
          );
        })}

        {[0, 0.25, 0.5, 0.75, 1.0].map((frac) => {
          const il = Math.min(Math.round(frac * (nL - 1)), nL - 1);
          const y = pad.top + (nL - 1 - il + 0.5) * cellH + 3;
          return (
            <text key={`l-${frac}`} x={pad.left - 8} y={y} textAnchor="end" fontSize="10" fill="#94a3b8">
              {lamGrid[il].toFixed(2)}
            </text>
          );
        })}

        <text x={W / 2} y={H - 3} textAnchor="middle" fontSize="11" fill="#cbd5e1">
          Cumulative Damage State h ∈ [0, H_crit]
        </text>
        <text
          x={14}
          y={H / 2}
          textAnchor="middle"
          fontSize="11"
          fill="#cbd5e1"
          transform={`rotate(-90 14 ${H / 2})`}
        >
          Hazard Intensity λ [h⁻¹]
        </text>
      </svg>
    </div>
  );
}

export default function WP2PolicyHeatmap({ solver }) {
  const [hovered, setHovered] = useState(null);
  if (!solver) return null;

  const fracs = solver.policy_region_fractions || {};

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.1rem' }}>
      <div className="panel-card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div className="legend-row">
          {Object.entries(ACTION_META).map(([k, meta]) => (
            <div className="legend-item" key={k}>
              <span className="legend-swatch" style={{ background: meta.color }} />
              <span>
                <strong>{meta.label}</strong> (
                {((fracs[meta.label] || 0) * 100).toFixed(1)}%)
              </span>
            </div>
          ))}
        </div>

        <div style={{ fontSize: '0.8rem', fontFamily: 'JetBrains Mono, monospace', color: '#38bdf8' }}>
          {hovered
            ? `State (h=${hovered.h.toFixed(2)}, λ=${hovered.lam.toFixed(2)}) → ${hovered.action} | V*=${hovered.value?.toFixed(1)}`
            : 'Hover any (h, λ) grid cell to inspect optimal robust action & value'}
        </div>
      </div>

      <div className="two-col">
        <PolicyGridSVG
          title="IMDP-PHM Minimax Policy Map a*(h, λ)"
          subtitle={`Worst-case Bellman over Θ • Converged in ${solver.outer_iterations} iter (GP R² = ${solver.gp_r2_score})`}
          hGrid={solver.h_grid}
          lamGrid={solver.lam_grid}
          policyMatrix={solver.imdp_policy_matrix}
          valueMatrix={solver.imdp_value_matrix}
          onHoverCell={setHovered}
        />

        <PolicyGridSVG
          title="Classical Point-Estimate MLE-MDP Policy Map"
          subtitle="Certainty-equivalent policy ignoring epistemic uncertainty bounds"
          hGrid={solver.h_grid}
          lamGrid={solver.lam_grid}
          policyMatrix={solver.mle_policy_matrix}
          valueMatrix={solver.mle_value_matrix}
          onHoverCell={setHovered}
        />
      </div>

      <div className="kpi-row">
        <div className="kpi-card">
          <div className="kpi-label">V*_IMDP(h₀, μ₀)</div>
          <div className="kpi-value" style={{ color: '#38bdf8' }}>{solver.v_imdp_pristine.toFixed(2)}</div>
          <div className="kpi-sub">Guaranteed worst-case cost bound</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">V*_MLE-MDP(h₀, μ₀)</div>
          <div className="kpi-value">{solver.v_mle_pristine.toFixed(2)}</div>
          <div className="kpi-sub">Optimistic point-estimate cost</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Epistemic Robustness Margin</div>
          <div className="kpi-value" style={{ color: '#fbbf24' }}>+{solver.epistemic_robustness_margin.toFixed(2)}</div>
          <div className="kpi-sub">Value of ambiguity protection</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Bellman Sup-Residual</div>
          <div className="kpi-value" style={{ color: '#34d399' }}>{solver.bellman_residual.toExponential(2)}</div>
          <div className="kpi-sub">Solved in {solver.solve_time_sec}s</div>
        </div>
      </div>
    </div>
  );
}
