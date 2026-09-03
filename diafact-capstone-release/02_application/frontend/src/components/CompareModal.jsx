// Side-by-side comparison modal: previous visit vs current visit
// Keys must match what /api/predict stores in input_features. `bmi` and
// `family_risk` are no longer inputs -- they are derived server-side -- so
// listing them here would render a permanent row of em-dashes.
const ROW_LABELS = {
  glucose: 'Glucose (mmol/L)', weight: 'Weight (kg)', height: 'Height (cm)',
  pulse_rate: 'Pulse Rate', systolic_bp: 'Systolic BP', diastolic_bp: 'Diastolic BP',
  hypertensive: 'Hypertension', cardiovascular_disease: 'Cardiovascular Disease',
  stroke: 'Previous Stroke', family_diabetes: 'Family: Diabetes',
  family_hypertension: 'Family: Hypertension',
};

const BINARY_KEYS = new Set(['hypertensive', 'cardiovascular_disease', 'stroke',
  'family_diabetes', 'family_hypertension']);

// For these features: lower = better (improvement)
const LOWER_IS_BETTER = new Set(['glucose', 'weight', 'systolic_bp', 'diastolic_bp', 'pulse_rate']);

function classify(key, prev, curr) {
  const p = parseFloat(prev), c = parseFloat(curr);
  if (isNaN(p) || isNaN(c) || Math.abs(p - c) < 0.01) return 'neutral';
  return LOWER_IS_BETTER.has(key) ? (c < p ? 'improved' : 'worsened') : (c > p ? 'improved' : 'worsened');
}

export default function CompareModal({ current, previous, patientName, onClose }) {
  const prevInputs = typeof previous.input_features === 'string'
    ? JSON.parse(previous.input_features) : previous.input_features;
  const currInputs = typeof current.input_features === 'string'
    ? JSON.parse(current.input_features) : current.input_features;

  const prevRisk = typeof previous === 'object' ? (previous.risk_percent ?? previous.risk_score * 100) : 0;
  const currRisk = current.risk_percent;
  const riskImproved = currRisk < prevRisk;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box" onClick={e => e.stopPropagation()}>

        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <div>
            <h2 style={{ fontSize: '1.2rem', fontWeight: 800 }}>Visit Comparison — {patientName}</h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.83rem', marginTop: '0.2rem' }}>
              Previous Visit #{previous.visit_number} vs Current Visit #{current.visit_number}
            </p>
          </div>
          <button onClick={onClose} style={{
            background: 'none', border: '1px solid var(--border)', borderRadius: '8px',
            color: 'var(--text-muted)', cursor: 'pointer', padding: '0.3rem 0.7rem', fontSize: '1rem',
          }}>✕</button>
        </div>

        {/* Risk summary row */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.5rem' }}>
          {[
            { label: 'Previous Risk', risk: prevRisk, visit: previous.visit_number },
            { label: 'Current Risk',  risk: currRisk,  visit: current.visit_number },
          ].map((item, i) => {
            const color = item.risk < 30 ? '#10b981' : item.risk < 60 ? '#f59e0b' : '#ef4444';
            return (
              <div key={i} style={{
                background: 'var(--surface2)', border: `1px solid ${color}44`,
                borderRadius: '10px', padding: '1rem', textAlign: 'center',
              }}>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '0.3rem' }}>
                  Visit #{item.visit} — {item.label}
                </div>
                <div style={{ fontSize: '2rem', fontWeight: 800, color }}>{item.risk.toFixed(1)}%</div>
              </div>
            );
          })}
        </div>

        {/* Overall change */}
        <div style={{
          background: riskImproved ? 'rgba(16,185,129,0.08)' : 'rgba(239, 68, 68, 0.08)',
          border: `1px solid ${riskImproved ? '#10b981' : '#ef4444'}44`,
          borderRadius: '8px', padding: '0.8rem 1rem', marginBottom: '1.5rem',
          color: riskImproved ? '#10b981' : '#ef4444', fontWeight: 700, fontSize: '0.95rem',
        }}>
          {riskImproved
            ? `✅ Risk improved by ${(prevRisk - currRisk).toFixed(1)}% since last visit`
            : `⚠️ Risk increased by ${(currRisk - prevRisk).toFixed(1)}% since last visit`}
        </div>

        {/* Feature-by-feature comparison */}
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', padding: '0.6rem 0.8rem', color: 'var(--text-muted)', fontSize: '0.78rem', textTransform: 'uppercase', borderBottom: '1px solid var(--border)' }}>Feature</th>
                <th style={{ textAlign: 'center', padding: '0.6rem 0.8rem', color: 'var(--text-muted)', fontSize: '0.78rem', textTransform: 'uppercase', borderBottom: '1px solid var(--border)' }}>Visit #{previous.visit_number}</th>
                <th style={{ textAlign: 'center', padding: '0.6rem 0.8rem', color: 'var(--text-muted)', fontSize: '0.78rem', textTransform: 'uppercase', borderBottom: '1px solid var(--border)' }}>Visit #{current.visit_number}</th>
                <th style={{ textAlign: 'center', padding: '0.6rem 0.8rem', color: 'var(--text-muted)', fontSize: '0.78rem', textTransform: 'uppercase', borderBottom: '1px solid var(--border)' }}>Change</th>
              </tr>
            </thead>
            <tbody>
              {Object.keys(ROW_LABELS).map(key => {
                const prev = prevInputs[key];
                const curr = currInputs[key];
                const cls  = classify(key, prev, curr);
                const diff = parseFloat(curr) - parseFloat(prev);
                const diffColor = cls === 'improved' ? '#10b981' : cls === 'worsened' ? '#ef4444' : 'var(--text-muted)';
                
                const isBinary = BINARY_KEYS.has(key);
                const formatVal = (v) => v === undefined || v === null ? '—' : (isBinary ? (v == 1 ? 'Yes' : 'No') : v);
                
                let diffStr = '—';
                if (!isNaN(diff) && diff !== 0) {
                  if (isBinary) diffStr = diff > 0 ? 'Added' : 'Removed';
                  else diffStr = `${diff > 0 ? '+' : ''}${diff.toFixed(1)}`;
                }

                return (
                  <tr key={key}>
                    <td style={{ padding: '0.6rem 0.8rem', fontSize: '0.88rem', color: 'var(--text-muted)', borderBottom: '1px solid var(--border)' }}>
                      {ROW_LABELS[key]}
                    </td>
                    <td style={{ textAlign: 'center', padding: '0.6rem 0.8rem', fontWeight: 600, borderBottom: '1px solid var(--border)' }}>{formatVal(prev)}</td>
                    <td style={{ textAlign: 'center', padding: '0.6rem 0.8rem', fontWeight: 700, borderBottom: '1px solid var(--border)', color: diffColor }}>{formatVal(curr)}</td>
                    <td style={{ textAlign: 'center', padding: '0.6rem 0.8rem', fontWeight: 600, borderBottom: '1px solid var(--border)', color: diffColor, fontSize: '0.85rem' }}>
                      {diffStr}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
