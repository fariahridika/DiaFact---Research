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
# PART 2 - PHYSICALLY-CONSISTENT COUNTERFACTUALS
# Addresses: derived features moved independently of their parents in 95-100%
#            of the published counterfactuals.
# Fix: DiCE searches ONLY raw actionable variables. Derived features are
#      recomputed from them at every prediction, so an incoherent profile
#      cannot be produced by construction.
# =============================================================================
print('\n' + '=' * 78)
print('PART 2: consistent counterfactual regeneration')
print('=' * 78)

import dice_ml
from dice_ml import Dice

RAW_ACTIONABLE = ['glucose', 'weight', 'systolic_bp', 'diastolic_bp',
                  'pulse_rate', 'hypertensive']


def rebuild(df_raw, age, height_m, cardio_offset, gender):
    """Reconstruct the full 16-feature frame from raw actionable variables."""
    X = pd.DataFrame(index=range(len(df_raw)), columns=FEATURES, dtype=float)
    dr = df_raw.reset_index(drop=True)
    for f in RAW_ACTIONABLE:
        X[f] = dr[f].astype(float).values
    X['age'] = age
    X['gender'] = gender
    X['bmi'] = X['weight'] / (height_m ** 2)
    X['bp_ratio'] = X['systolic_bp'] / (X['diastolic_bp'] + 1)
    X['pulse_pressure'] = X['systolic_bp'] - X['diastolic_bp']
    X['bmi_age'] = X['bmi'] * age
    X['glucose_bmi'] = X['glucose'] * X['bmi']
    X['cardio_risk'] = X['hypertensive'] + cardio_offset
    for f in FEATURES:
        if X[f].isna().all():
            X[f] = 0.0
    return X[FEATURES].astype(float)


class ConsistentModel:
    """Wraps the classifier so derived features are always recomputed."""
    classes_ = np.array([0, 1])

    def __init__(self, mdl, age, h, off, gender):
        self.m, self.age, self.h, self.off, self.gender = mdl, age, h, off, gender

    def fit(self, X, y):
        return self

    def _full(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=RAW_ACTIONABLE)
        return rebuild(X, self.age, self.h, self.off, self.gender)

    def predict(self, X):
        return self.m.predict(self._full(X))

    def predict_proba(self, X):
        return self.m.predict_proba(self._full(X))


model = joblib.load(BASE + 'models/diahealth_xgboost.pkl')
Xf = X_test[FEATURES].astype(float)
y_prob = model.predict_proba(Xf)[:, 1]
y_pred = model.predict(Xf)
tp = np.where((y_test == 1) & (y_pred == 1))[0]
sel = tp[np.argsort(y_prob[tp])[::-1][:20]]
print('patients:', list(sel))

# SHAP importance aggregated back onto the raw parents
PARENTS = {'bmi_age': ['weight'], 'glucose_bmi': ['glucose', 'weight'],
           'bmi': ['weight'], 'bp_ratio': ['systolic_bp', 'diastolic_bp'],
           'pulse_pressure': ['systolic_bp', 'diastolic_bp'],
           'cardio_risk': ['hypertensive']}


def raw_shap(i):
    s = pd.Series(np.abs(SV[min(int(i), len(SV) - 1)]), index=FEATURES)
    agg = {r: 0.0 for r in RAW_ACTIONABLE}
    for f, v in s.items():
        if f in agg:
            agg[f] += float(v)
        elif f in PARENTS:
            for r in PARENTS[f]:
                if r in agg:
                    agg[r] += float(v) / len(PARENTS[f])
    return agg


def shap_top_raw(i, k=3):
    return [f for f, _ in sorted(raw_shap(i).items(), key=lambda kv: -kv[1])[:k]]


train_raw = X_train[RAW_ACTIONABLE].astype(float).copy()
train_raw['diabetic'] = y_train
CONT = [c for c in RAW_ACTIONABLE if c != 'hypertensive']

results = {}
for tag, kfun in [('standard_consistent', None),
                  ('shap_guided_consistent', lambda i: shap_top_raw(i, 3))]:
    allcf = []
    for i in sel:
        o = Xf.iloc[int(i)]
        h = float(np.sqrt(o['weight'] / o['bmi'])) if o['bmi'] > 0 else 1.6
        off = float(o['cardio_risk'] - o['hypertensive'])
        cm = ConsistentModel(model, float(o['age']), h, off, float(o['gender']))
        d = dice_ml.Data(dataframe=train_raw, continuous_features=CONT,
                         outcome_name='diabetic')
        exp = Dice(d, dice_ml.Model(model=cm, backend='sklearn'), method='genetic')
        vary = RAW_ACTIONABLE if kfun is None else kfun(int(i))
        try:
            r = exp.generate_counterfactuals(
                X_test[RAW_ACTIONABLE].astype(float).iloc[[int(i)]],
                total_CFs=3, desired_class='opposite', features_to_vary=vary)
            c = r.cf_examples_list[0].final_cfs_df.copy()
            c['instance_idx'] = int(i)
            allcf.append(c)
        except Exception as e:
            print('  patient %s failed: %s' % (i, e))
    results[tag] = pd.concat(allcf, ignore_index=True) if allcf else pd.DataFrame()
    print('%s: %d CFs' % (tag, len(results[tag])))
    results[tag].to_csv(OUT + 'cf_%s.csv' % tag, index=False)


def concentration(o_raw, cf_raw, i):
    """Sparsity-NEUTRAL alignment: SHAP mass captured, relative to the most
    that ANY counterfactual changing that many features could capture."""
    agg = raw_shap(i)
    ch = [f for f in RAW_ACTIONABLE if abs(float(cf_raw[f]) - float(o_raw[f])) > 1e-6]
    if not ch:
        return np.nan
    got = sum(agg[f] for f in ch)
    best = sum(sorted(agg.values(), reverse=True)[:len(ch)])
    return got / best if best else np.nan


rows = []
for tag, df in results.items():
    if not len(df):
        continue
    for i in df['instance_idx'].unique():
        o = X_test[RAW_ACTIONABLE].astype(float).iloc[int(i)]
        of = Xf.iloc[int(i)]
        h = float(np.sqrt(of['weight'] / of['bmi']))
        off = float(of['cardio_risk'] - of['hypertensive'])
        for _, r in df[df['instance_idx'] == i].iterrows():
            full = rebuild(pd.DataFrame([r[RAW_ACTIONABLE]]), float(of['age']),
                           h, off, float(of['gender']))
            ch = [f for f in RAW_ACTIONABLE if abs(float(r[f]) - float(o[f])) > 1e-6]
            rows.append(dict(
                method=tag, idx=int(i),
                valid=int(model.predict(full)[0] == 0),
                proximity_l1=sum(abs(float(r[f]) - float(o[f])) for f in RAW_ACTIONABLE),
                sparsity=len(ch) / len(RAW_ACTIONABLE),
                concentration=concentration(o, r, i)))

M = pd.DataFrame(rows)
M.to_csv(OUT + 'cf_metrics_consistent.csv', index=False)
summ = M.groupby('method').agg(validity=('valid', 'mean'),
                               proximity=('proximity_l1', 'mean'),
                               sparsity=('sparsity', 'mean'),
                               alignment=('concentration', 'mean'),
                               n=('valid', 'size'))
print('\n### Consistent-CF results ###')
print(summ.round(4).to_string())
summ.to_csv(OUT + 'table_cf_consistent.csv')

print('\n### Paired tests (n=20 patients) ###')
for col in ['proximity_l1', 'sparsity', 'concentration']:
    g = M.groupby(['method', 'idx'])[col].mean().reset_index()
    a = g[g.method == 'standard_consistent'].sort_values('idx')[col].values
    b = g[g.method == 'shap_guided_consistent'].sort_values('idx')[col].values
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ok = n > 5 and not (np.isnan(a).any() or np.isnan(b).any())
    if ok:
        st, p = wilcoxon(a, b)
        d = a - b
        ci = np.percentile([np.mean(rng.choice(d, len(d), replace=True))
                            for _ in range(20000)], [2.5, 97.5])
        print('  %-14s std=%.4f shap=%.4f diff=%+.4f 95%%CI[%+.4f,%+.4f] W=%.1f p=%.5f'
              % (col, a.mean(), b.mean(), d.mean(), ci[0], ci[1], st, p))
    rho, pp = spearmanr(M['sparsity'], M[col], nan_policy='omit')
    print('     confound check: rho(sparsity, %s) = %+.3f (p=%.3g)' % (col, rho, pp))

print('\nDONE. Outputs written to', OUT)
