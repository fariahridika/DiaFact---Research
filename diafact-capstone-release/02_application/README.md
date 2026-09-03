# DiaFact

Clinical decision-support prototype for Type 2 diabetes risk: calibrated
prediction, exact SHAP attribution, and consistency-safe SHAP-guided
counterfactual recommendations.

Implements the method from *"A Prescriptive and Explainable Machine Learning
Framework for Type 2 Diabetes Risk Prediction and Personalized Counterfactual
Recommendations."*

> **Decision support only.** Every number here is a statement about a model, not
> a diagnosis, and the recommendations are model-level targets that have not
> been validated against clinical outcomes.

---

## Quick start

**Prerequisites:** MySQL (XAMPP is fine), Node 18+, Python 3.10+.

```bash
# 1. Database
#    Import diafact_schema.sql via phpMyAdmin, or:
mysql -u root < diafact_schema.sql
```

```bash
# 2. Build the model artifacts from the published experiment (once)
cd ml_service
pip install -r requirements.txt
python prepare_artifacts.py --source "<path to>/Final Results/exp_v2"
```

```bash
# 3. Run everything (Windows)
start.bat
```

Or start the three services by hand:

```bash
cd ml_service && python app.py          # :5001
cd backend    && npm install && node index.js   # :3001
cd frontend   && npm install && npm run dev     # :5173
```

Seed 100 real DiaHealth patients through the real prediction path:

```bash
cd backend && npm run seed -- --csv "<path>/DiaHealth_Diabetes Dataset.csv"
```

---

## Architecture

```
React (5173) → Node/Express (3001) → Flask ML (5001)
                      ↓
                MySQL (3306)
```

`ml_service/clinical.py` is the single source of truth for feature formulas,
validation ranges, and clinical constraints.

---

## The model

XGBoost on 16 Boruta-selected features from DiaHealth (5,437 Bangladeshi
patients, 6.34% prevalence). `prepare_artifacts.py` refuses to build unless the
model reproduces the paper's published test metrics exactly:

| | AUC-ROC | F1 | Precision | Recall |
|---|---|---|---|---|
| Published (Table 1, @0.5) | 0.8203 | 0.3766 | 0.3412 | 0.4203 |
| This service | 0.8203 | 0.3766 | 0.3412 | 0.4203 |

### Calibration

The model was trained on BorderlineSMOTE-resampled data balanced 50/50, so its
raw output is inflated relative to real prevalence — the case-study patient
scores 87.7% raw against a calibrated 43.8%. Showing the raw number to a
patient as "your risk" is a safety problem, so an isotonic calibrator fitted on
out-of-fold predictions is applied. Test Brier improves 0.0711 → 0.0527.
Calibration is monotonic, so no ranking and no AUC changes.

### Operating points

There is no single correct cutoff at 6.3% prevalence, so three are offered and
the expected precision is shown in the UI rather than hidden. Thresholds are
chosen on out-of-fold data; the figures below are held-out test estimates.

| Setting | Threshold | Precision | Recall | Flags |
|---|---|---|---|---|
| Screening | 0.080 | 0.180 | 0.783 | 27.6% |
| **Balanced** (default) | **0.174** | **0.357** | **0.362** | **6.4%** |
| Confirmatory | 0.348 | 0.406 | 0.188 | 2.9% |

Even at the best setting most flagged patients are false positives. That is a
property of the problem, and the interface says so instead of announcing a
diagnosis.

---

## Recommendations

Counterfactuals are generated under three constraints:

1. **Raw variables only.** DiCE searches `glucose`, `weight`, `systolic_bp`,
   `diastolic_bp`, `pulse_rate`. Every engineered feature (`bmi`, `bp_ratio`,
   `pulse_pressure`, `bmi_age`, `glucose_bmi`, `cardio_risk`) is recomputed
   from them before each prediction, so a profile that loses weight while
   gaining BMI cannot be produced.
2. **`hypertensive` is frozen.** It records a diagnosis; setting it to 0 is not
   an action a patient can take.
3. **Direction-constrained.** A recommendation may only move a variable the way
   that lowers clinical risk, within clinician-plausible bounds — never below
   BMI 18.5 for weight, never above the patient's current value for glucose,
   blood pressure or pulse.

Both strategies are returned so the contribution is visible: **SHAP-guided**
(top 3 variables by parent-credited |SHAP|) against **unconstrained** (all
five). Measured over 40 real high-risk test patients:

| | Mean changes | Mean alignment | Direction violations |
|---|---|---|---|
| Unconstrained | 1.83 | 0.802 | 0 |
| SHAP-guided | 1.34 | 0.911 | 0 |

The paper reports direction violations in 75% of unconstrained and 52% of
SHAP-guided counterfactuals; constraint 3 removes them.

---

## Security

- Parameterised SQL throughout; ids validated as positive integers.
- Every clinical value bounded before it can reach the model.
- Helmet security headers, explicit CORS allowlist, 60 req/min rate limit,
  32 KB body cap.
- Optional `x-api-key` gate (`API_KEY` in `.env`), constant-time compared.
- Internal errors are logged, never returned to the caller.
- Both services bind `127.0.0.1` by default.
- No pickles on the serving path — the model loads as XGBoost JSON and the
  calibrator as interpolation knots, so loading an artifact cannot execute code.
- Seeding is a CLI script, not an unauthenticated HTTP endpoint.

Copy `backend/.env.example` to `backend/.env` and set `DB_PASS` and `API_KEY`
before exposing this beyond localhost. The service refuses to start in
`NODE_ENV=production` with a blank database password.

---

## Repository notes

`ml_service/_quarantine_wrong_dataset/` holds the artifacts this app previously
shipped. They were not built from DiaHealth (12.19% prevalence vs 6.33%, mean
weight 75 kg vs 53.6 kg, 0.1% of rows matching the source CSV) and scored
AUC 0.674 / precision 0.111 on the real test split. Kept for reproducibility;
nothing loads them. See the README in that folder.
