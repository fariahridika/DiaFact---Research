'use strict';

const express = require('express');
const router  = express.Router();
const db      = require('../db');
const { validateIdentity, parseId } = require('../validate');

/**
 * GET /api/patients — paginated list, newest first.
 *
 * The previous version selected MAX(v.risk_label), which is a lexicographic
 * maximum over the strings 'High' and 'Low': any patient with a single 'Low'
 * visit displayed as 'Low' forever, regardless of their latest result. The
 * label and percent now come from the genuinely latest visit.
 */
router.get('/', async (req, res, next) => {
  const limit  = Math.min(Math.max(Number(req.query.limit) || 50, 1), 200);
  const offset = Math.max(Number(req.query.offset) || 0, 0);
  const q      = typeof req.query.q === 'string' ? req.query.q.trim().slice(0, 100) : '';

  try {
    const params = [];
    let where = '';
    if (q) {
      // Escape LIKE wildcards so a search for "100%" is not a prefix scan.
      where = 'WHERE p.name LIKE ? ESCAPE \'\\\\\'';
      params.push(`%${q.replace(/[\\%_]/g, m => '\\' + m)}%`);
    }

    const [rows] = await db.query(
      `SELECT p.id, p.name, p.age, p.gender, p.created_at,
              COUNT(v.id) AS visit_count,
              MAX(v.visited_at) AS last_visit,
              latest.risk_label   AS last_risk_label,
              latest.risk_percent AS last_risk_percent
         FROM patients p
         LEFT JOIN patient_visits v ON v.patient_id = p.id
         LEFT JOIN (
              SELECT pv.patient_id, pv.risk_label, pv.risk_percent
                FROM patient_visits pv
                JOIN (SELECT patient_id, MAX(visit_number) AS mx
                        FROM patient_visits GROUP BY patient_id) m
                  ON m.patient_id = pv.patient_id AND m.mx = pv.visit_number
         ) latest ON latest.patient_id = p.id
         ${where}
        GROUP BY p.id, p.name, p.age, p.gender, p.created_at,
                 latest.risk_label, latest.risk_percent
        ORDER BY p.created_at DESC, p.id DESC
        LIMIT ? OFFSET ?`,
      [...params, limit, offset]
    );
    res.json(rows);
  } catch (err) {
    next(err);
  }
});

// GET /api/patients/:id — one patient with all visits
router.get('/:id', async (req, res, next) => {
  const id = parseId(req.params.id);
  if (id === undefined || id === null) {
    return res.status(400).json({ error: 'id must be a positive integer' });
  }
  try {
    const [[patient]] = await db.query(
      'SELECT id, name, age, gender, created_at FROM patients WHERE id = ?', [id]
    );
    if (!patient) return res.status(404).json({ error: 'Patient not found' });

    const [visits] = await db.query(
      `SELECT id, visit_number, input_features, model_features,
              risk_score, risk_label, risk_percent,
              shap_values, counterfactuals, visited_at
         FROM patient_visits
        WHERE patient_id = ?
        ORDER BY visit_number ASC, id ASC`,
      [id]
    );
    res.json({ ...patient, visits });
  } catch (err) {
    next(err);
  }
});

// POST /api/patients — create a patient record
router.post('/', async (req, res, next) => {
  const { value, errors } = validateIdentity(req.body, { required: true });
  if (errors.length) {
    return res.status(400).json({ error: 'Validation failed', details: errors });
  }
  try {
    const [result] = await db.query(
      'INSERT INTO patients (name, age, gender) VALUES (?, ?, ?)',
      [value.name, value.age, value.gender]
    );
    res.status(201).json({ id: result.insertId, ...value });
  } catch (err) {
    next(err);
  }
});

module.exports = router;
