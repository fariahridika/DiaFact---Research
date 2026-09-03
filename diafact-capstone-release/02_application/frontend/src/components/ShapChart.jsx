import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';

const LABELS = {
  age: 'Age', gender: 'Gender', pulse_rate: 'Pulse Rate',
  systolic_bp: 'Systolic BP', diastolic_bp: 'Diastolic BP',
  glucose: 'Glucose', weight: 'Weight', bmi: 'BMI',
  hypertensive: 'Hypertensive', bp_ratio: 'BP Ratio',
  pulse_pressure: 'Pulse Pressure', bmi_age: 'BMI×Age',
  glucose_bmi: 'Glucose×BMI', cardio_risk: 'Cardio Risk',
  family_hypertension: 'Family Hypertension', family_risk: 'Family Risk',
};

export default function ShapChart({ shapValues }) {
  // Sort by absolute value, top 10
  const data = Object.entries(shapValues)
    .map(([k, v]) => ({ feature: LABELS[k] || k, value: v, abs: Math.abs(v) }))
    .sort((a, b) => b.abs - a.abs)
    .slice(0, 10);

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    const d = payload[0].payload;
    const dir = d.value > 0 ? '⚠️ Increases' : '✅ Decreases';
    const color = d.value > 0 ? 'var(--danger)' : 'var(--success)';
    
    return (
      <div style={{
        background: 'var(--surface)', border: '1px solid var(--border)',
        padding: '0.8rem 1rem', borderRadius: '8px', fontSize: '0.9rem',
        boxShadow: 'var(--shadow)',
      }}>
        <div style={{ fontWeight: 700, marginBottom: '0.4rem', color: 'var(--text)' }}>
          Factor: {d.feature}
        </div>
        <div style={{ color: color, fontWeight: 600 }}>
          {dir} risk 
        </div>
      </div>
    );
  };

  return (
    <div style={{ width: '100%', height: 350 }}>
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ left: 10, right: 30 }}>
          <XAxis type="number" tick={{ fill: 'var(--text-muted)', fontSize: 12 }} />
          <YAxis type="category" dataKey="feature" width={130}
            tick={{ fill: 'var(--text)', fontSize: 13, fontWeight: 500 }} />
          <Tooltip content={<CustomTooltip />} cursor={{fill: 'var(--surface2)'}} />
          <ReferenceLine x={0} stroke="var(--border)" strokeWidth={2} />
          <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={18}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.value > 0 ? 'var(--danger)' : 'var(--success)'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
