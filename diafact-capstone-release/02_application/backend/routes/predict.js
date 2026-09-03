'use strict';

const express = require('express');
const router  = express.Router();
const db      = require('../db');
const ml      = require('../mlClient');
const { validateClinical, validateIdentity, parseId } = require('../validate');

/**
 * POST /api/predict
 * Body: { patient_id? , name?, age?, gender, ...clinical, operating_point? }
 */
router.post('/', async (req, res, next) => {
  const pid = parseId(req.body.patient_id);
  if (pid === undefined) {
    return res.status(400).json({ error: 'patient_id must be a positive integer' });
  }

  const clinical = validateClinical(req.body);
  const identity = validateIdentity(req.body, { required: pid === null });
  const errors = [...clinical.errors, ...identity.errors];
  if (errors.length) {
    return res.status(400).json({ error: 'Validation failed', details: errors });
  }

  // Acquiring the connection must be inside the try: Express 4 does not catch
  // rejections from async handlers, so a database outage here would surface as
  // an unhandled rejection and take the whole process down.
  let conn;
  try {
    conn = await db.getConnection();
    let patientId = pid;

    if (patientId === null) {
      const [r] = await conn.query(
        'INSERT INTO patients (name, age, gender) VALUES (?, ?, ?)',
        [identity.value.name, identity.value.age, identity.value.gender]
      );
      patientId = r.insertId;
    } else {
      const [[p]] = await conn.query('SELECT id FROM patients WHERE id = ?', [patientId]);
      if (!p) return res.status(404).json({ error: `Patient ${patientId} not found` });
    }

    // Everything the model needs, all validated. `height` is sent so the ML
    // service can derive BMI itself and keep weight and BMI coherent when it
    // generates counterfactuals.
    const c = clinical.value;
    const mlPayload = {
      age: c.age, gender: c.gender,
      pulse_rate: c.pulse_rate, systolic_bp: c.systolic_bp, diastolic_bp: c.diastolic_bp,
      glucose: c.glucose, weight: c.weight, height: c.height,
      hypertensive: c.hypertensive,
      family_diabetes: c.family_diabetes,
      family_hypertension: c.family_hypertension,
      cardiovascular_disease: c.cardiovascular_disease,
      stroke: c.stroke,
    };
    if (c.operating_point) mlPayload.operating_point = c.operating_point;

    let mlResult;
    try {
      mlResult = await ml.predict(mlPayload);
    } catch (e) {
      if (e.code === 'ECONNREFUSED' || e.code === 'ECONNABORTED') {
        return res.status(503).json({
          error: 'Scoring service unavailable. Start ml_service/app.py first.',
        });
      }
      if (e.response && e.response.status === 400) {
        return res.status(400).json({
          error: 'Validation failed at scoring service',
          details: e.response.data && e.response.data.details,
        });
      }
      throw e;
    }

    // Visit numbering inside a transaction. Counting rows outside one lets two
    // concurrent visits for the same patient both become "visit 2".
    await conn.beginTransaction();
    let insertId, visitNumber;
    try {
      const [[row]] = await conn.query(
        'SELECT COALESCE(MAX(visit_number), 0) AS n FROM patient_visits '
        + 'WHERE patient_id = ? FOR UPDATE', [patientId]
      );
      visitNumber = Number(row.n) + 1;

      const [ins] = await conn.query(
        `INSERT INTO patient_visits
          (patient_id, visit_number, input_features, model_features,
           risk_score, risk_label, risk_percent, shap_values, counterfactuals)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [
          patientId, visitNumber,
          JSON.stringify(mlPayload),
          JSON.stringify(mlResult.model_features),
          mlResult.risk_score, mlResult.risk_label, mlResult.risk_percent,
          JSON.stringify(mlResult.shap_values),
          JSON.stringify({
            shap_guided: mlResult.counterfactuals || [],
            unconstrained: mlResult.counterfactuals_unconstrained || [],
            diagnostics: mlResult.cf_diagnostics || {},
          }),
        ]
      );
      insertId = ins.insertId;
      await conn.commit();
    } catch (e) {
      await conn.rollback();
      throw e;
    }

    const [[patient]] = await conn.query(
      'SELECT id, name, age, gender, created_at FROM patients WHERE id = ?', [patientId]
    );

    // Order by visit_number, not visited_at: DATETIME has one-second resolution,
    // so two visits in the same second order arbitrarily.
    const [prev] = await conn.query(
      `SELECT id, visit_number, input_features, model_features, risk_score,
              risk_label, risk_percent, shap_values, counterfactuals, visited_at
         FROM patient_visits
        WHERE patient_id = ? AND id <> ?
        ORDER BY visit_number DESC, id DESC LIMIT 1`,
      [patientId, insertId]
    );

    res.json({
      patient,
      visit: {
        id: insertId,
        visit_number: visitNumber,
        input_features: mlPayload,
        model_features: mlResult.model_features,
        risk_score: mlResult.risk_score,
        risk_label: mlResult.risk_label,
        risk_percent: mlResult.risk_percent,
        risk_raw_uncalibrated: mlResult.risk_raw_uncalibrated,
        operating_point: mlResult.operating_point,
        shap_values: mlResult.shap_values,
        shap_ranking: mlResult.shap_ranking,
        counterfactuals: mlResult.counterfactuals,
        counterfactuals_unconstrained: mlResult.counterfactuals_unconstrained,
        cf_diagnostics: mlResult.cf_diagnostics,
        disclaimer: mlResult.disclaimer,
        visited_at: new Date().toISOString(),
      },
      has_previous: prev.length > 0,
      previous_visit: prev[0] || null,
      is_new_patient: visitNumber === 1,
    });
  } catch (err) {
    if (err && (err.code === 'ECONNREFUSED' || err.code === 'PROTOCOL_CONNECTION_LOST'
                || err.code === 'ER_ACCESS_DENIED_ERROR' || err.fatal)) {
      console.error('Database unavailable:', err.code);
      return res.status(503).json({ error: 'Database unavailable. Is MySQL running?' });
    }
    next(err);
  } finally {
    if (conn) conn.release();
  }
});

// GET /api/predict/visits/:visitId
router.get('/visits/:visitId', async (req, res, next) => {
  const id = parseId(req.params.visitId);
  if (id === undefined || id === null) {
    return res.status(400).json({ error: 'visitId must be a positive integer' });
  }
  try {
    const [[visit]] = await db.query(
      'SELECT * FROM patient_visits WHERE id = ?', [id]
    );
    if (!visit) return res.status(404).json({ error: 'Visit not found' });
    res.json(visit);
  } catch (err) {
    next(err);
  }
});

module.exports = router;
