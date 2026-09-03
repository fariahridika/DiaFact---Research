const LABELS = {
  glucose: 'Glucose', weight: 'Weight',
  pulse_rate: 'Pulse Rate', systolic_bp: 'Systolic BP', diastolic_bp: 'Diastolic BP',
};

const UNITS = {
  glucose: 'mmol/L', weight: 'kg',
  pulse_rate: 'bpm', systolic_bp: 'mmHg', diastolic_bp: 'mmHg',
};

const EFFORT_COLOR = {
  Easy: 'var(--success)', Moderate: 'var(--warning)', Hard: 'var(--danger)',
};

function DiffTarget({ feat, orig, newVal }) {
  const o = Number(orig);
  const diff = Number(newVal) - o;
  const unit = UNITS[feat] ? ` ${UNITS[feat]}` : '';
  const label = LABELS[feat] || feat;
  const known = Number.isFinite(o);

  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '0.8rem 1rem', background: 'var(--surface)',
      border: '1px solid var(--border)', borderRadius: '8px', marginBottom: '0.5rem',
    }}>
      <div style={{ fontWeight: 600, color: 'var(--text)' }}>Target {label}</div>
      <div style={{ textAlign: 'right' }}>
        <div style={{ fontWeight: 800, fontSize: '1.1rem', color: 'var(--success)' }}>
          {newVal}{unit}
        </div>
        {known && (
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Currently {o}{unit} ({diff < 0 ? 'decrease' : 'increase'} by {Math.abs(diff).toFixed(1)})
          </div>
        )}
      </div>
    </div>
  );
}

export default function CFCards({ counterfactuals, originalInputs, riskLabel, threshold }) {
  if (!counterfactuals?.length) {
    if (riskLabel === 'High') {
      return (
        <div className="alert-banner alert-warning" style={{
          background: 'rgba(245, 158, 11, 0.1)',
          border: '1px solid rgba(245, 158, 11, 0.3)', color: '#b45309',
        }}>
          ⚠️ No recommendation could be generated within safe clinical limits.
          Reaching a lower risk band would require changes beyond what the search
          is permitted to suggest. Please refer to a physician for an individual plan.
        </div>
      );
    }
    return (
      <div className="alert-banner alert-success">
        ✅ This patient is below the flagging threshold. No intervention targets generated.
      </div>
    );
  }

  const pct = threshold != null ? `${(threshold * 100).toFixed(1)}%` : 'the threshold';

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '1.5rem' }}>
      {counterfactuals.map((cf) => (
        <div key={cf.cf_index} style={{
          background: 'var(--surface2)', border: '1px solid var(--border)',
          borderRadius: '12px', padding: '1.5rem', boxShadow: 'var(--shadow)',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.2rem' }}>
            <span style={{ fontWeight: 800, fontSize: '1.1rem', color: 'var(--primary)' }}>
              Action Plan {cf.cf_index}
            </span>
            <span style={{
              padding: '0.3rem 0.8rem', borderRadius: '999px',
              background: `color-mix(in srgb, ${EFFORT_COLOR[cf.effort]} 15%, transparent)`,
              color: EFFORT_COLOR[cf.effort], fontSize: '0.8rem', fontWeight: 700,
              border: `1px solid color-mix(in srgb, ${EFFORT_COLOR[cf.effort]} 30%, transparent)`,
            }}>
              {cf.n_changes} change{cf.n_changes === 1 ? '' : 's'} · {cf.effort}
            </span>
          </div>

          <div style={{ marginBottom: '1.5rem' }}>
            {Object.entries(cf.changes).map(([feat, newVal]) => (
              <DiffTarget key={feat} feat={feat} orig={originalInputs?.[feat]} newVal={newVal} />
            ))}
          </div>

          <div style={{
            background: 'color-mix(in srgb, var(--success) 10%, transparent)',
            border: '1px solid color-mix(in srgb, var(--success) 30%, transparent)',
            borderRadius: '8px', padding: '1rem', textAlign: 'center',
          }}>
            <div style={{
              fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.2rem',
              textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600,
            }}>
              Projected Risk
            </div>
            <div style={{ color: 'var(--success)', fontWeight: 800, fontSize: '1.5rem' }}>
              {cf.new_risk_percent}%
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--success)', marginTop: '0.2rem', fontWeight: 600 }}>
              Below the {pct} flagging threshold
            </div>
            {cf.alignment != null && (
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                SHAP alignment {cf.alignment.toFixed(2)}
                {cf.alignment >= 0.999 ? ' — targets this patient’s top drivers exactly' : ''}
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
