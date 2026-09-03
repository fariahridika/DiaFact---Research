import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { runPredict, getPatient } from '../api';

/**
 * Demo presets drawn from the DiaHealth cohort the model was trained on:
 * Bangladeshi adults, mean weight ~54 kg and mean BMI ~22.5. The previous
 * presets (98 kg / BMI 33.9) described a population the model never saw.
 */
const PRESETS = {
  'High Risk': {
    age: 60, gender: 'Female', pulse_rate: 98, systolic_bp: 130, diastolic_bp: 86,
    glucose: 11.3, weight: 56, height: 165,
    hypertensive: 1, family_diabetes: 0, family_hypertension: 0,
    cardiovascular_disease: 0, stroke: 0,
  },
  'Low Risk': {
    age: 28, gender: 'Female', pulse_rate: 72, systolic_bp: 112, diastolic_bp: 74,
    glucose: 4.9, weight: 48, height: 155,
    hypertensive: 0, family_diabetes: 0, family_hypertension: 0,
    cardiovascular_disease: 0, stroke: 0,
  },
};

const OPERATING_POINTS = [
  { key: 'screening',    label: 'Screening',    hint: 'Catches most cases, many false alarms' },
  { key: 'balanced',     label: 'Balanced',     hint: 'Default — even trade-off' },
  { key: 'confirmatory', label: 'Confirmatory', hint: 'Few false alarms, misses more' },
];

const MEASUREMENTS = [
  { name: 'pulse_rate',   label: 'Pulse Rate (bpm)',       placeholder: '60–100',  step: '1'   },
  { name: 'systolic_bp',  label: 'Systolic BP (mmHg)',     placeholder: '90–180',  step: '1'   },
  { name: 'diastolic_bp', label: 'Diastolic BP (mmHg)',    placeholder: '60–110',  step: '1'   },
  { name: 'glucose',      label: 'Fasting Glucose (mmol/L)', placeholder: '3.9–25', step: '0.1' },
  { name: 'weight',       label: 'Weight (kg)',            placeholder: '30–150',  step: '0.1' },
  { name: 'height',       label: 'Height (cm)',            placeholder: '140–200', step: '0.1' },
];

// `family_risk` and `cardio_risk` are NOT collected: they are composites the
// server derives (family_diabetes + family_hypertension, and hypertensive +
// cardiovascular_disease + stroke). Collecting them separately is what made the
// old build feed the model values it was never trained on.
const HISTORY = [
  { name: 'hypertensive',           label: 'Diagnosed Hypertension' },
  { name: 'cardiovascular_disease', label: 'Cardiovascular Disease' },
  { name: 'stroke',                 label: 'Previous Stroke' },
  { name: 'family_diabetes',        label: 'Family History: Diabetes' },
  { name: 'family_hypertension',    label: 'Family History: Hypertension' },
];

const defaultForm = {
  patient_id: '', name: '', age: '', gender: 'Female',
  pulse_rate: '', systolic_bp: '', diastolic_bp: '',
  glucose: '', weight: '', height: '',
  hypertensive: 0, cardiovascular_disease: 0, stroke: 0,
  family_diabetes: 0, family_hypertension: 0,
  operating_point: 'balanced',
};

export default function PatientForm() {
  const [form, setForm]       = useState(defaultForm);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState('');
  const [details, setDetails] = useState([]);
  const navigate = useNavigate();

  // BMI is derived, never entered, and never sent: the server computes it from
  // weight and height so weight and BMI can never disagree.
  const bmi = (() => {
    const w = parseFloat(form.weight), h = parseFloat(form.height) / 100;
    if (!w || !h || h <= 0) return null;
    const v = w / (h * h);
    return Number.isFinite(v) ? v : null;
  })();

  const handleChange = (e) => setForm(f => ({ ...f, [e.target.name]: e.target.value }));

  const handlePatientIdBlur = async () => {
    if (!form.patient_id) return;
    try {
      const { data } = await getPatient(form.patient_id);
      if (!data) return;
      let last = {};
      if (data.visits?.length) {
        let inputs = data.visits[data.visits.length - 1].input_features;
        if (typeof inputs === 'string') inputs = JSON.parse(inputs);
        if (inputs) last = inputs;
      }
      setForm(f => ({
        ...f, ...last,
        gender: data.gender ?? (last.gender === 1 ? 'Male' : 'Female'),
        name: data.name || f.name,
        age: data.age || f.age,
        patient_id: f.patient_id,
      }));
    } catch {
      /* unknown id: leave the form as typed */
    }
  };

  const applyPreset = (key) => setForm(f => ({ ...f, ...PRESETS[key] }));

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(''); setDetails([]); setLoading(true);
    try {
      const { data } = await runPredict(form);
      navigate('/results', { state: data });
    } catch (err) {
      const r = err.response?.data;
      setError(r?.error || 'Connection failed. Is the backend running?');
      setDetails(Array.isArray(r?.details) ? r.details : []);
    } finally {
      setLoading(false);
    }
  };

  const bmiFlag = bmi !== null && (bmi < 12 || bmi > 60);

  return (
    <div style={{ maxWidth: 720, margin: '0 auto' }}>
      <div style={{ marginBottom: '1.5rem' }}>
        <h1 style={{ fontSize: '1.6rem', fontWeight: 800, marginBottom: '0.3rem' }}>
          T2D Risk Assessment
        </h1>
        <p style={{ color: 'var(--text-muted)' }}>
          Enter patient clinical data for a calibrated risk estimate, SHAP explanation,
          and SHAP-guided intervention targets.
        </p>
      </div>

      <div style={{ display: 'flex', gap: '0.7rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
        <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem', alignSelf: 'center' }}>
          Demo presets:
        </span>
        {Object.keys(PRESETS).map(k => (
          <button key={k} type="button" className="btn btn-outline"
            style={{ padding: '0.4rem 0.9rem', fontSize: '0.85rem' }}
            onClick={() => applyPreset(k)}>
            {k === 'High Risk' ? '🔴' : '🟢'} {k} Patient
          </button>
        ))}
      </div>

      <form className="card" onSubmit={handleSubmit}>
        <div className="section-title">Returning Patient? (Optional)</div>
        <div style={{ marginBottom: '1.5rem' }}>
          <div className="form-group">
            <label>Patient ID (leave blank for new patient)</label>
            <input type="number" name="patient_id" placeholder="e.g. 42" min="1"
              value={form.patient_id} onChange={handleChange} onBlur={handlePatientIdBlur} />
          </div>
        </div>

        <div className="section-title">Patient Information</div>
        <div className="form-grid" style={{ marginBottom: '1.5rem' }}>
          <div className="form-group">
            <label>Full Name *</label>
            <input name="name" required maxLength={150} placeholder="Patient name"
              value={form.name} onChange={handleChange} />
          </div>
          <div className="form-group">
            <label>Age (years) *</label>
            <input type="number" name="age" min="1" max="120" required
              placeholder="e.g. 45" value={form.age} onChange={handleChange} />
          </div>
          <div className="form-group">
            <label>Gender *</label>
            <div className="toggle-group">
              {['Male', 'Female'].map(g => (
                <button type="button" key={g}
                  className={`toggle-btn${form.gender === g ? ' active' : ''}`}
                  onClick={() => setForm(f => ({ ...f, gender: g }))}>{g}</button>
              ))}
            </div>
          </div>
        </div>

        <div className="section-title">Clinical Measurements</div>
        <div className="form-grid" style={{ marginBottom: '0.5rem' }}>
          {MEASUREMENTS.map(f => (
            <div key={f.name} className="form-group">
              <label>{f.label} *</label>
              <input type="number" step={f.step} name={f.name} required
                placeholder={f.placeholder} value={form[f.name]} onChange={handleChange} />
            </div>
          ))}
          <div className="form-group">
            <label>BMI (derived)</label>
            <input readOnly value={bmi === null ? '' : bmi.toFixed(1)}
              placeholder="From weight & height"
              style={{ background: 'var(--surface2)', cursor: 'not-allowed' }} />
          </div>
        </div>
        {bmiFlag && (
          <div style={{ color: 'var(--danger)', fontSize: '0.85rem', marginBottom: '1rem' }}>
            ⚠️ BMI of {bmi.toFixed(1)} is outside the plausible range — check weight and height.
          </div>
        )}

        <div className="section-title" style={{ marginTop: '1rem' }}>Medical History</div>
        <div className="form-grid" style={{ marginBottom: '1.5rem' }}>
          {HISTORY.map(f => (
            <div key={f.name} className="form-group">
              <label>{f.label}</label>
              <div className="toggle-group">
                {['No', 'Yes'].map((opt, i) => (
                  <button type="button" key={opt}
                    className={`toggle-btn${form[f.name] === i ? ' active' : ''}`}
                    onClick={() => setForm(p => ({ ...p, [f.name]: i }))}>{opt}</button>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="section-title">Operating Point</div>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '0.7rem' }}>
          Sets how readily the model flags a patient. There is no free lunch here —
          catching more cases always means more false alarms.
        </p>
        <div className="toggle-group" style={{ marginBottom: '2rem', flexWrap: 'wrap' }}>
          {OPERATING_POINTS.map(op => (
            <button type="button" key={op.key} title={op.hint}
              className={`toggle-btn${form.operating_point === op.key ? ' active' : ''}`}
              onClick={() => setForm(f => ({ ...f, operating_point: op.key }))}>
              {op.label}
            </button>
          ))}
        </div>

        {error && (
          <div className="alert-banner alert-danger"
            style={{ marginBottom: '1rem', padding: '0.8rem 1rem', display: 'block' }}>
            ⚠️ {error}
            {details.length > 0 && (
              <ul style={{ margin: '0.5rem 0 0 1.1rem', fontSize: '0.85rem' }}>
                {details.map((d, i) => <li key={i}>{d}</li>)}
              </ul>
            )}
          </div>
        )}

        <button type="submit" className="btn btn-primary" disabled={loading}
          style={{ width: '100%', padding: '0.9rem', fontSize: '1rem' }}>
          {loading ? '⏳ Analysing…' : '🔍 Run T2D Assessment'}
        </button>
      </form>
    </div>
  );
}
