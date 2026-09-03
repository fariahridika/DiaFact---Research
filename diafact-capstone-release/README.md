# DiaFact — Prescriptive and Explainable ML for Type 2 Diabetes Risk

Code, data preparation scripts and figure generators for the capstone project
*"A Prescriptive and Explainable Machine Learning Framework for Type 2 Diabetes
Risk Prediction and Personalised Counterfactual Recommendations."*

**Repository:** `<PASTE YOUR GITHUB URL HERE>`

**Authors:** Faria Alam Hridika, Nosrat Jahan Mohime, Mashrat Alam Mahi
**Supervisor:** Md. Ziaur Rahman
Department of CSE, Bangladesh Army International University of Science and
Technology (BAIUST), Cumilla, Bangladesh.

> **Decision support only.** Everything here is a statement about a model, not a
> diagnosis. The recommendations are model-level targets and have not been
> validated against clinical outcomes.

---

## What this project does

Most diabetes risk models stop at a probability. This one goes one step further
and computes a short list of changes that would move a patient below the risk
threshold — constrained so that the advice is physically possible, clinically
safe, and targeted at the variables that actually drive *that* patient's risk.

Two ideas carry the work:

1. **Consistency-safe generation.** The counterfactual search ranges over five
   raw clinical variables only. Every engineered feature (BMI, `bp_ratio`,
   `pulse_pressure`, `bmi_age`, `glucose_bmi`, `cardio_risk`) is recomputed from
   them before each prediction, so a profile like "lose weight while gaining BMI
   at constant height" cannot be expressed.

2. **A sparsity-neutral alignment score.** Measures whether a recommendation
   targets a patient's highest-impact variables, normalised so it cannot be
   inflated by changing more variables or fewer.

---

## Repository layout

```
01_research/      the experiments behind the results
  notebooks/        pre-processing, model, xai, cf, summary
  scripts/          reproducibility re-runs (threshold tuning, consistent CFs)
  results/          result tables as CSV, including frozen_hyp/
02_application/   DiaFact, the three-tier web application
  ml_service/       Flask: model, calibration, SHAP, consistency-safe DiCE
  backend/          Node/Express: validation, persistence, security
  frontend/         React: assessment, results, history, visit comparison
03_figures/       scripts that regenerate every figure in the book
04_data/          the DiaHealth dataset
docs/             REVIEW_FIXES.md — the internal audit that reshaped the project
```

---

## Quick start

### 1. Reproduce the research

```bash
cd 01_research/notebooks
# run in order:
#   pre-processing.ipynb -> model.ipynb -> xai.ipynb -> cf.ipynb -> summary.ipynb
```

Each notebook writes a checkpoint, so a later stage can be re-run without
repeating the earlier ones.

### 2. Run the application

Prerequisites: MySQL (XAMPP is fine), Node 18+, Python 3.10+.

```bash
mysql -u root < 02_application/diafact_schema.sql
```

```bash
cd 02_application/ml_service
pip install -r requirements.txt
python prepare_artifacts.py --source "<path to>/Final Results/exp_v2"
python app.py                       # :5001
```

```bash
cd 02_application/backend
cp .env.example .env                # set DB_PASS and API_KEY before any shared use
npm install && node index.js        # :3001
```

```bash
cd 02_application/frontend
npm install && npm run dev          # :5173
```

On Windows, `02_application/start.bat` does all of the above.

Load 100 real DiaHealth patients through the real prediction path:

```bash
cd 02_application/backend
npm run seed -- --csv "../../04_data/DiaHealth_Diabetes Dataset.csv"
```

### 3. Regenerate the figures

```bash
cd 03_figures/scripts
python figs1.py && python figs2.py && python figs3.py && python figs4.py
python gantt.py
python eval_app.py                  # canonical deployed-system evaluation
```

Paths at the top of each script point at the experiment folder; adjust them if
you move things.

---

## Headline results

| Quantity | Value |
|---|---|
| Dataset | 5,437 patients, 344 positive (6.33%) |
| Split | 4,349 train / 1,088 test, stratified |
| Features | 16 of 20, selected by Boruta |
| XGBoost (served) | AUC-ROC 0.8203, F1 0.3766, precision 0.3412 |
| AutoGluon (best classifier) | AUC-ROC 0.8858 |
| Precision at matched recall | AutoGluon 0.5577 vs XGBoost 0.3452 |
| Brier, before → after calibration | 0.0711 → 0.0527 |
| Proximity reduction (SHAP-guided) | −41.4%, *p* = 10⁻⁵ |
| Sparsity reduction | −36.3%, *p* = 9×10⁻⁵ |
| Alignment gain | +5.7%, *p* = 0.010 |
| Direction violations, research pipeline | 75% / 52% |
| **Direction violations, deployed system** | **0 of 304 plans** |

### Why XGBoost, when AutoGluon scores higher

AutoGluon wins on every threshold-independent metric. It is used for the
comparison and not for serving, because its `best_quality` preset is a two-layer
stack of dozens of models including a neural network, and SHAP's TreeExplainer
cannot compute exact Shapley values over that. The prescriptive stage depends on
exact attribution, so we trade roughly 0.21 precision at matched recall for it —
stated openly rather than hidden.

---

## Reproducibility

Three checks run automatically and fail loudly rather than silently:

- **Model provenance.** `prepare_artifacts.py` aborts unless the rebuilt model
  reproduces AUC-ROC 0.8203, F1 0.3766, precision 0.3412 and recall 0.4203 to
  within 5×10⁻³. This check exists because an earlier build shipped a model
  trained on a different dataset entirely (precision 0.111 on real DiaHealth).
- **Serving path.** The saved artefact must match the sklearn prediction path to
  within 10⁻⁶. Measured difference: 0.
- **Feature reconstruction.** Rebuilding all 16 features for all 1,088 test
  patients through the application's own code must match the training matrix.
  Measured maximum difference: 9.09×10⁻¹³.

Environment: Python 3.12, seed 42. Key versions in
`02_application/ml_service/requirements.txt`.

---

## Security notes

The application stores patient records. Implemented: parameterised SQL, bounded
input validation, Helmet headers, CORS allowlist, rate limiting, 32 KB body cap,
optional constant-time API key, loopback binding, and **no pickle files on the
serving path** (the model loads as XGBoost JSON, the calibrator as interpolation
knots).

Not implemented, and required before any real deployment: encryption at rest and
an access audit log.

Copy `backend/.env.example` to `backend/.env` and set `DB_PASS` and `API_KEY`
before exposing this beyond localhost.

---

## A note on `docs/REVIEW_FIXES.md`

That file is the internal audit that reshaped this project. It documents the
point at which we found our own headline alignment metric was mathematically
broken — it rewarded changing more variables, so a counterfactual changing
everything scored a perfect 1.0 — and the finding it had produced pointed in the
wrong direction.

It is included deliberately. The corrected result (+5.7%) is much smaller than
the one a flattering metric would have given us (+34%), and the reasoning behind
choosing the smaller number is the most transferable part of this work.

---

## Data and licensing

The DiaHealth dataset is published by Prama, Zaman, Sarker and Mamun
(Mendeley Data, doi:10.17632/7m7555vgrn.1) for research use. It is included here
for convenience; please cite the original authors.

Code released for academic use. All libraries used are open source.

---

## Citation

```bibtex
@mastersthesis{diafact2026,
  title  = {A Prescriptive and Explainable Machine Learning Framework for
            Type 2 Diabetes Risk Prediction and Personalised Counterfactual
            Recommendations},
  author = {Hridika, Faria Alam and Mohime, Nosrat Jahan and Mahi, Mashrat Alam},
  school = {Bangladesh Army International University of Science and Technology},
  year   = {2026},
  note   = {Capstone Project, Department of CSE. Supervisor: Md. Ziaur Rahman}
}
```
