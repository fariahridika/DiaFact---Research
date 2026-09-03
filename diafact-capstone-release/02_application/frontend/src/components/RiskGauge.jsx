// Circular arc risk gauge
export default function RiskGauge({ percent }) {
  const r      = 70;
  const cx     = 90;
  const cy     = 90;
  const stroke = 12;
  const circ   = Math.PI * r;  // half-circle arc
  const pct    = Math.max(0, Math.min(100, percent));
  const offset = circ - (pct / 100) * circ;

  const color = pct < 30 ? 'var(--success)' : pct < 60 ? 'var(--warning)' : 'var(--danger)';
  const label = pct < 30 ? 'Low Risk' : pct < 60 ? 'Moderate Risk' : 'High Risk';

  return (
    <div style={{ textAlign: 'center' }}>
      <svg viewBox="0 0 180 100" width="220" style={{ overflow: 'visible' }}>
        {/* Track */}
        <path
          d={`M${cx - r},${cy} A${r},${r} 0 0,1 ${cx + r},${cy}`}
          fill="none" stroke="var(--border)" strokeWidth={stroke} strokeLinecap="round"
        />
        {/* Value arc */}
        <path
          d={`M${cx - r},${cy} A${r},${r} 0 0,1 ${cx + r},${cy}`}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={`${circ}`}
          strokeDashoffset={offset}
          style={{ transition: 'stroke-dashoffset 1s ease, stroke 0.5s' }}
        />
        {/* Percent label */}
        <text x={cx} y={cy - 8} textAnchor="middle" fontSize="30" fontWeight="800" fill="var(--text)">
          {pct.toFixed(1)}%
        </text>
        {/* Sub label */}
        <text x={cx} y={cy + 16} textAnchor="middle" fontSize="12" fontWeight="600" fill="var(--text-muted)">
          Risk Probability
        </text>
      </svg>
      <div style={{
        display: 'inline-block',
        padding: '0.4rem 1.5rem',
        borderRadius: '999px',
        fontWeight: 700,
        fontSize: '1.05rem',
        background: `color-mix(in srgb, ${color} 15%, transparent)`,
        color,
        border: `1.5px solid color-mix(in srgb, ${color} 30%, transparent)`,
        marginTop: '0.5rem',
      }}>
        {label}
      </div>
    </div>
  );
}
