'use strict';

/**
 * Input validation for DiaFact.
 *
 * The previous version passed `parseFloat(undefined)` -> NaN straight through
 * to the model, which happily scored it. Every clinical value is bounded here
 * before it can reach the ML service, and the bounds match ml_service/clinical.py.
 */

const RANGES = {
  age:          [1, 120, 'years'],
  pulse_rate:   [25, 220, 'bpm'],
  systolic_bp:  [60, 260, 'mmHg'],
  diastolic_bp: [30, 160, 'mmHg'],
  glucose:      [1, 40, 'mmol/L'],
  weight:       [20, 300, 'kg'],
  height:       [100, 230, 'cm'],
};

const FLAGS = [
  'hypertensive', 'family_diabetes', 'family_hypertension',
  'cardiovascular_disease', 'stroke',
];

const OPERATING_POINTS = ['screening', 'balanced', 'confirmatory'];

function validateClinical(body) {
  const errors = [];
  const out = {};

  for (const [key, [lo, hi, unit]] of Object.entries(RANGES)) {
    const raw = body[key];
    if (raw === undefined || raw === null || raw === '') {
      errors.push(`${key} is required`);
      continue;
    }
    const v = Number(raw);
    if (!Number.isFinite(v)) {
      errors.push(`${key} must be a number`);
      continue;
    }
    if (v < lo || v > hi) {
      errors.push(`${key}=${v} is outside the plausible range ${lo}-${hi} ${unit}`);
      continue;
    }
    out[key] = v;
  }

  // Gender accepted as a label from the UI, or 0/1 from an API client.
  const g = body.gender;
  if (g === 'Male' || g === 1 || g === '1') out.gender = 1;
  else if (g === 'Female' || g === 0 || g === '0') out.gender = 0;
  else errors.push("gender must be 'Male' or 'Female'");

  for (const f of FLAGS) {
    const v = Number(body[f] ?? 0);
    if (![0, 1].includes(v)) { errors.push(`${f} must be 0 or 1`); continue; }
    out[f] = v;
  }

  if (out.systolic_bp !== undefined && out.diastolic_bp !== undefined
      && out.systolic_bp <= out.diastolic_bp) {
    errors.push(`systolic_bp (${out.systolic_bp}) must exceed diastolic_bp (${out.diastolic_bp})`);
  }

  if (out.weight !== undefined && out.height !== undefined) {
    const bmi = out.weight / Math.pow(out.height / 100, 2);
    if (bmi < 8 || bmi > 80) {
      errors.push(`weight/height give an implausible BMI of ${bmi.toFixed(1)}`);
    }
  }

  if (body.operating_point !== undefined && body.operating_point !== null
      && body.operating_point !== '') {
    if (!OPERATING_POINTS.includes(body.operating_point)) {
      errors.push(`operating_point must be one of ${OPERATING_POINTS.join(', ')}`);
    } else {
      out.operating_point = body.operating_point;
    }
  }

  return { value: out, errors };
}

function validateIdentity(body, { required }) {
  const errors = [];
  const out = {};

  const name = typeof body.name === 'string' ? body.name.trim() : '';
  if (name) {
    if (name.length > 150) errors.push('name must be 150 characters or fewer');
    else out.name = name;
  } else if (required) {
    errors.push('name is required for a new patient');
  }

  if (body.age !== undefined && body.age !== '') {
    const a = Number(body.age);
    if (!Number.isInteger(a) || a < 1 || a > 120) errors.push('age must be a whole number 1-120');
    else out.age = a;
  } else if (required) {
    errors.push('age is required for a new patient');
  }

  if (body.gender === 'Male' || body.gender === 'Female') out.gender = body.gender;
  else if (required) errors.push("gender must be 'Male' or 'Female'");

  return { value: out, errors };
}

/** Positive integer id, or null. Rejects "1 OR 1=1", "1.5", "-3", "abc". */
function parseId(raw) {
  if (raw === undefined || raw === null || raw === '') return null;
  const s = String(raw).trim();
  if (!/^\d{1,10}$/.test(s)) return undefined; // undefined signals "invalid"
  const n = Number(s);
  return n >= 1 ? n : undefined;
}

module.exports = { validateClinical, validateIdentity, parseId, OPERATING_POINTS };
