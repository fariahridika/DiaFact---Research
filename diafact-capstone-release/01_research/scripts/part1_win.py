# =============================================================================
# TEHI 2026 - REVIEWER-RESPONSE RERUN
# Run on Kaggle (Python 3.11) with the exp_v2 outputs attached as a dataset.
#
# Produces:
#   PART 1 - threshold-tuned model comparison INCLUDING AutoGluon
#   PART 2 - physically-consistent counterfactuals + sparsity-neutral alignment
#
# Requirements on Kaggle:  pip install dice-ml
#   (autogluon, xgboost, lightgbm, catboost are preinstalled on Kaggle images)
# =============================================================================
import os, warnings, joblib, numpy as np, pandas as pd
warnings.filterwarnings('ignore')

# --- portability shim: AutoGluon artefacts were pickled on Linux ---
import pathlib, platform
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

BASE = 'D:/thesis paper/TEHI 2026/exp_v2/exp_v2/Final Results/exp_v2/'
OUT  = 'D:/thesis paper/TEHI 2026/rerun_outputs/'
os.makedirs(OUT, exist_ok=True)

from sklearn.metrics import (f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score, brier_score_loss, accuracy_score)
from scipy.stats import wilcoxon, spearmanr
rng = np.random.default_rng(42)

splits  = joblib.load(BASE + 'checkpoints/nb01_splits.pkl')
X_test,  y_test  = splits['X_test'],  splits['y_test'].values
X_train, y_train = splits['X_train'], splits['y_train'].values
xai      = joblib.load(BASE + 'results/xai/xai_results.pkl')
SV       = xai['shap_values']
FEATURES = list(xai['X_test_shap'].columns)
MODIFIABLE = [f for f in FEATURES if f not in
              ['age', 'gender', 'family_diabetes', 'family_hypertension',
               'family_risk', 'cardiovascular_disease', 'stroke']]
print('features', len(FEATURES), '| modifiable', len(MODIFIABLE))


# =============================================================================
# PART 1 - THRESHOLD-TUNED MODEL COMPARISON
# Addresses: "XGBoost was chosen on F1 at an untuned 0.5 threshold"
# =============================================================================
print('\n' + '=' * 78)
print('PART 1: threshold sweep')
print('=' * 78)

probs = {}
for name, path in [('XGBoost',  'models/diahealth_xgboost.pkl'),
                   ('LightGBM', 'models/diahealth_lightgbm.pkl'),
                   ('CatBoost', 'models/diahealth_catboost.pkl')]:
    probs[name] = joblib.load(BASE + path).predict_proba(X_test[FEATURES].astype(float))[:, 1]

from autogluon.tabular import TabularPredictor
ag = TabularPredictor.load(BASE + 'models/autogluon_diahealth', require_version_match=False)
pa = ag.predict_proba(X_test[FEATURES].astype(float))
probs['AutoGluon'] = pa[1].values if 1 in pa.columns else pa.iloc[:, -1].values

grid = np.linspace(0.01, 0.99, 197)
rows = []
for n, p in probs.items():
    bt = max(grid, key=lambda t: f1_score(y_test, (p >= t).astype(int), zero_division=0))
    for tag, t in [('@0.50 (published)', 0.5), ('@F1-optimal', bt)]:
        q = (p >= t).astype(int)
        rows.append(dict(
            model=n, setting=tag, threshold=round(float(t), 3),
            precision=precision_score(y_test, q, zero_division=0),
            recall=recall_score(y_test, q),
            f1=f1_score(y_test, q, zero_division=0),
            accuracy=accuracy_score(y_test, q),
            auc_roc=roc_auc_score(y_test, p),
            auc_pr=average_precision_score(y_test, p),
            brier=brier_score_loss(y_test, p)))
T = pd.DataFrame(rows)
T.to_csv(OUT + 'table_threshold_tuned.csv', index=False)
print(T.round(4).to_string(index=False))

xr = recall_score(y_test, (probs['XGBoost'] >= 0.5).astype(int))
print('\nRecall-matched to XGBoost@0.5 (recall=%.4f):' % xr)
rm = []
for n, p in probs.items():
    cand = [t for t in grid if recall_score(y_test, (p >= t).astype(int)) >= xr]
    if cand:
        t = max(cand)
        q = (p >= t).astype(int)
        rm.append(dict(model=n, threshold=round(float(t), 3),
                       recall=recall_score(y_test, q),
                       precision=precision_score(y_test, q, zero_division=0),
                       f1=f1_score(y_test, q, zero_division=0)))
RM = pd.DataFrame(rm)
RM.to_csv(OUT + 'table_recall_matched.csv', index=False)
print(RM.round(4).to_string(index=False))
np.save(OUT + 'probs.npy', probs, allow_pickle=True)


