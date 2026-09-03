"""Canonical evaluation of the DEPLOYED system, run once so the book and the
figures quote identical numbers. Saves per-counterfactual records to CSV."""
import sys, json, warnings, joblib
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, r"D:/thesis paper/DiaFact App (3)/DiaFact App/ml_service")
import app as A
from clinical import RAW_ACTIONABLE, violates_direction

BASE = r"D:/thesis paper/TEHI 2026/exp_v2/exp_v2/Final Results/exp_v2/"
DATA = r"D:/thesis paper/Capstone_Project_Template_for_Final_Book/figures/_data/"
N_PATIENTS = 60

splits = joblib.load(BASE + "checkpoints/nb01_splits.pkl")
X = splits["X_test"]; y = splits["y_test"].astype(int).values
client = A.app.test_client()


def payload(i):
    o = X.iloc[i]
    h = float(np.sqrt(o["weight"] / o["bmi"]) * 100) if o["bmi"] > 0 else 160.0
    return dict(age=float(o["age"]), gender=int(o["gender"]), pulse_rate=float(o["pulse_rate"]),
                systolic_bp=float(o["systolic_bp"]), diastolic_bp=float(o["diastolic_bp"]),
                glucose=float(o["glucose"]), weight=float(o["weight"]), height=round(h, 1),
                hypertensive=int(o["hypertensive"]), family_diabetes=int(o["family_diabetes"]),
                family_hypertension=int(o["family_hypertension"]),
                cardiovascular_disease=int(o["cardiovascular_disease"]), stroke=int(o["stroke"]))


p_cal = A.calibrate(A.raw_proba(X))
order = np.argsort(-p_cal)[:N_PATIENTS]
print(f"scoring the {N_PATIENTS} highest-risk test patients")

recs, cov = [], {"standard": [0, 0], "shap_guided": [0, 0]}
latency = []
for n, i in enumerate(order, 1):
    r = client.post("/predict", json=payload(int(i)))
    assert r.status_code == 200, r.status_code
    d = r.get_json()
    if d["risk_label"] != "High":
        continue
    orig = {f: float(d["model_features"][f]) for f in RAW_ACTIONABLE}
    for strat, key in [("standard", "counterfactuals_unconstrained"),
                       ("shap_guided", "counterfactuals")]:
        lst = d[key]
        cov[strat][1] += 1
        if lst:
            cov[strat][0] += 1
        for cf in lst:
            prop = dict(orig); prop.update(cf["changes"])
            recs.append(dict(patient=int(i), strategy=strat, n_changes=cf["n_changes"],
                             alignment=cf["alignment"], new_risk=cf["new_risk_score"],
                             risk_before=d["risk_score"],
                             direction_violation=int(violates_direction(orig, prop)),
                             changed=";".join(sorted(cf["changes"]))))
    for s in ("standard", "shap_guided"):
        k = "cf_diagnostics"
        if k in d and s in d[k] and "seconds" in d[k][s]:
            latency.append(d[k][s]["seconds"])
    if n % 15 == 0:
        print(f"  {n}/{len(order)}")

df = pd.DataFrame(recs)
df.to_csv(DATA + "app_cf_records.csv", index=False)

summary = {}
for s in ("standard", "shap_guided"):
    sub = df[df.strategy == s]
    summary[s] = dict(
        n_cfs=int(len(sub)),
        mean_changes=float(sub.n_changes.mean()),
        mean_alignment=float(sub.alignment.mean()),
        pct_single_change=float((sub.n_changes == 1).mean() * 100),
        direction_violations=int(sub.direction_violation.sum()),
        coverage_patients=f"{cov[s][0]}/{cov[s][1]}",
        coverage_pct=float(cov[s][0] / max(cov[s][1], 1) * 100),
        mean_risk_after=float(sub.new_risk.mean() * 100),
    )
summary["n_high_risk_patients"] = int(cov["standard"][1])
summary["mean_search_seconds"] = float(np.mean(latency)) if latency else None
summary["mean_risk_before"] = float(df.risk_before.mean() * 100)

from scipy import stats
pp = df.groupby(["strategy", "patient"])[["n_changes", "alignment"]].mean().reset_index()
a = pp[pp.strategy == "standard"].set_index("patient")
b = pp[pp.strategy == "shap_guided"].set_index("patient")
com = a.index.intersection(b.index)
for col in ["n_changes", "alignment"]:
    summary[f"p_{col}"] = float(stats.wilcoxon(a.loc[com, col], b.loc[com, col])[1])
summary["n_paired_patients"] = int(len(com))

json.dump(summary, open(DATA + "app_eval.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
