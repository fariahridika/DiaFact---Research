'use strict';

/**
 * Seed the database with real DiaHealth patients, scored through the real
 * prediction path.
 *
 * This replaces the previous POST /api/seed route, which was a problem on
 * several counts: it was unauthenticated (anyone reachable could stuff the
 * database), it read a hard-coded `m:/thesis/...` path that exists on one
 * machine, it invented American names for a Bangladeshi cohort, and it wrote
 * placeholder visits with risk_score 0 and empty SHAP. It also tested
 * `obj.Diabetic` when selecting rows but `patient.diabetic` when labelling
 * them, so every seeded visit came out 'Low'.
 *
 * Usage:
 *   node scripts/seed.js --csv "<path>/DiaHealth_Diabetes Dataset.csv" [--n 100]
 */

require('dotenv').config();

const fs   = require('fs');
const path = require('path');

const API   = process.env.SEED_API || `http://127.0.0.1:${process.env.PORT || 3001}/api`;
const KEY   = process.env.API_KEY || '';

function arg(flag, dflt) {
  const i = process.argv.indexOf(flag);
  return i > -1 && process.argv[i + 1] ? process.argv[i + 1] : dflt;
}

const CSV = arg('--csv', path.resolve(__dirname,
  '../../../../TEHI 2026/exp_v2/exp_v2/DiaHealth_Diabetes Dataset.csv'));
const WANT = Number(arg('--n', 100));

/** Minimal RFC-4180-ish parser: handles quoted fields containing commas. */
function parseCsv(text) {
  const rows = [];
  let field = '', row = [], inQ = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQ) {
      if (ch === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; } else inQ = false;
      } else field += ch;
    } else if (ch === '"') inQ = true;
    else if (ch === ',') { row.push(field); field = ''; }
    else if (ch === '\n') { row.push(field); rows.push(row); row = []; field = ''; }
    else if (ch !== '\r') field += ch;
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  const header = rows.shift().map(h => h.trim());
  return rows
    .filter(r => r.length === header.length)
    .map(r => Object.fromEntries(header.map((h, i) => [h, r[i].trim()])));
}

// Bangladeshi names, matching the cohort the model was actually trained on.
const GIVEN_F = ['Ayesha', 'Fatima', 'Nusrat', 'Rima', 'Shirin', 'Taslima', 'Rokeya',
  'Nasrin', 'Sultana', 'Momtaz', 'Jharna', 'Parveen', 'Rehana', 'Shahida'];
const GIVEN_M = ['Rafiq', 'Kamal', 'Jashim', 'Anwar', 'Habib', 'Sohel', 'Mizanur',
  'Faruk', 'Delwar', 'Shafiq', 'Nazrul', 'Bashir', 'Tanvir', 'Alamgir'];
const FAMILY = ['Rahman', 'Islam', 'Hossain', 'Ahmed', 'Chowdhury', 'Khan', 'Uddin',
  'Akter', 'Begum', 'Sarker', 'Mia', 'Haque', 'Ali', 'Siddiqui'];

const pick = a => a[Math.floor(Math.random() * a.length)];

async function post(url, body) {
  const headers = { 'Content-Type': 'application/json' };
  if (KEY) headers['x-api-key'] = KEY;
  const res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body) });
  const data = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, data };
}

(async () => {
  if (!fs.existsSync(CSV)) {
    console.error(`CSV not found: ${CSV}\nPass one with --csv "<path>"`);
    process.exit(1);
  }
  console.log(`reading ${CSV}`);
  const rows = parseCsv(fs.readFileSync(CSV, 'utf8'));
  console.log(`${rows.length} rows`);

  // Balance positives and negatives so the demo shows both outcomes.
  const pos = rows.filter(r => r.diabetic === 'Yes');
  const neg = rows.filter(r => r.diabetic === 'No');
  const half = Math.floor(WANT / 2);
  const shuffle = a => a.map(v => [Math.random(), v]).sort((x, y) => x[0] - y[0]).map(v => v[1]);
  const chosen = shuffle([...shuffle(pos).slice(0, half), ...shuffle(neg).slice(0, WANT - half)]);

  let created = 0, rejected = 0, failed = 0;
  const rejectReasons = new Map();

  for (const r of chosen) {
    const gender = r.gender === 'Male' ? 'Male' : 'Female';
    const body = {
      name: `${pick(gender === 'Male' ? GIVEN_M : GIVEN_F)} ${pick(FAMILY)}`,
      age: Number(r.age),
      gender,
      pulse_rate: Number(r.pulse_rate),
      systolic_bp: Number(r.systolic_bp),
      diastolic_bp: Number(r.diastolic_bp),
      glucose: Number(r.glucose),
      weight: Number(r.weight),
      height: Number(r.height) * 100,          // dataset stores metres
      hypertensive: Number(r.hypertensive) || 0,
      family_diabetes: Number(r.family_diabetes) || 0,
      family_hypertension: Number(r.family_hypertension) || 0,
      cardiovascular_disease: Number(r.cardiovascular_disease) || 0,
      stroke: Number(r.stroke) || 0,
    };

    const { ok, status, data } = await post(`${API}/predict`, body);
    if (ok) { created++; continue; }
    if (status === 400) {
      // DiaHealth contains physiologically impossible rows (weight 3 kg,
      // BMI 574, glucose 0). Validation rejecting them is correct behaviour.
      rejected++;
      for (const d of (data.details || ['unspecified'])) {
        const kind = String(d).split(/[= ]/)[0];
        rejectReasons.set(kind, (rejectReasons.get(kind) || 0) + 1);
      }
    } else {
      failed++;
      if (failed <= 3) console.error(`  HTTP ${status}: ${JSON.stringify(data).slice(0, 200)}`);
    }
  }

  console.log(`\nseeded  : ${created}`);
  console.log(`rejected: ${rejected} (implausible source rows -- expected)`);
  if (rejectReasons.size) {
    console.log('  by field: ' + [...rejectReasons.entries()]
      .sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k}=${v}`).join(', '));
  }
  console.log(`errors  : ${failed}`);
})().catch(e => { console.error(e); process.exit(1); });
