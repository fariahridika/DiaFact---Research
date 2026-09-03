"""Generate the data-driven figures for the capstone book. All numbers come
from the saved experiment artifacts -- nothing here is hand-entered."""
import os, json, warnings, joblib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
warnings.filterwarnings("ignore")

BASE = r"D:/thesis paper/TEHI 2026/exp_v2/exp_v2/Final Results/exp_v2/"
RERUN = r"D:/thesis paper/TEHI 2026/rerun_outputs/"
APP = r"D:/thesis paper/DiaFact App (3)/DiaFact App/ml_service/artifacts/"
OUT = r"D:/thesis paper/Capstone_Project_Template_for_Final_Book/figures/"
os.makedirs(OUT, exist_ok=True)

# ---- consistent house style -------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 8.8,
    "xtick.labelsize": 8.8,
    "ytick.labelsize": 8.8,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
TW = 6.2                                   # \textwidth in inches
C = {"xgb": "#2166ac", "lgb": "#4393c3", "cat": "#92c5de", "ag": "#b2182b",
     "std": "#8c8c8c", "shap": "#1a7f5a", "acc": "#d6604d", "ok": "#1a7f5a"}

def save(fig, name):
    fig.savefig(OUT + name)
    plt.close(fig)
    print("  saved", name)

# ---- load -------------------------------------------------------------------
splits = joblib.load(BASE + "checkpoints/nb01_splits.pkl")
X_tr, y_tr = splits["X_train"], splits["y_train"].astype(int)
X_te, y_te = splits["X_test"], splits["y_test"].astype(int).values
xai = joblib.load(BASE + "results/xai/xai_results.pkl")
SV, Xs = np.asarray(xai["shap_values"]), xai["X_test_shap"]
FEAT = list(Xs.columns)
meta = json.load(open(APP + "meta.json"))
probs = np.load(RERUN + "probs.npy", allow_pickle=True).item() \
    if os.path.exists(RERUN + "probs.npy") else None

# probs.npy holds the four models' test-set probabilities from the verified
# rerun; using them avoids unpickling models across library versions.
MODELS = {k: np.asarray(v) for k, v in probs.items()}
print("models:", list(MODELS))

# =============================================================== FIG: dataset
raw = pd.read_csv(r"D:/thesis paper/TEHI 2026/exp_v2/exp_v2/DiaHealth_Diabetes Dataset.csv")
raw["y"] = (raw.diabetic == "Yes").astype(int)
fig, ax = plt.subplots(1, 3, figsize=(TW, 2.05))
n0, n1 = (raw.y == 0).sum(), (raw.y == 1).sum()
b = ax[0].bar(["Non-diabetic", "Diabetic"], [n0, n1], color=["#b8d4e3", C["acc"]],
              edgecolor="black", linewidth=0.5, width=0.55)
ax[0].bar_label(b, fmt="%d", fontsize=8.5, padding=2)
ax[0].set_ylabel("Patients"); ax[0].set_title("(a) Class balance")
ax[0].set_ylim(0, n0 * 1.16)
ax[0].text(0.5, 0.80, f"{n1/len(raw)*100:.2f}% positive", transform=ax[0].transAxes,
           ha="center", fontsize=8.5, style="italic")

for lab, sub, col in [("Non-diabetic", raw[raw.y == 0], "#b8d4e3"), ("Diabetic", raw[raw.y == 1], C["acc"])]:
    ax[1].hist(sub.glucose.clip(2, 20), bins=38, density=True, alpha=0.65, color=col, label=lab)
ax[1].set_xlabel("Fasting glucose (mmol/L)"); ax[1].set_ylabel("Density")
ax[1].set_title("(b) Glucose by class"); ax[1].legend(loc="upper right")

ax[2].hist(raw.age, bins=32, color="#7fb3d5", edgecolor="black", linewidth=0.3)
ax[2].set_xlabel("Age (years)"); ax[2].set_ylabel("Patients"); ax[2].set_title("(c) Age distribution")
fig.tight_layout(pad=0.6)
save(fig, "fig_dataset_overview.png")

# ====================================================== FIG: model comparison
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                             average_precision_score, f1_score, precision_score,
                             recall_score, brier_score_loss)
order = ["AutoGluon", "XGBoost", "LightGBM", "CatBoost"]
order = [m for m in order if m in MODELS]
cols = {"AutoGluon": C["ag"], "XGBoost": C["xgb"], "LightGBM": C["lgb"], "CatBoost": C["cat"]}

fig, ax = plt.subplots(1, 2, figsize=(TW, 2.55))
for m in order:
    p = MODELS[m]
    fpr, tpr, _ = roc_curve(y_te, p)
    ax[0].plot(fpr, tpr, lw=1.5, color=cols[m], label=f"{m} ({roc_auc_score(y_te,p):.3f})")
    pr, rc, _ = precision_recall_curve(y_te, p)
    ax[1].plot(rc, pr, lw=1.5, color=cols[m], label=f"{m} ({average_precision_score(y_te,p):.3f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
ax[0].set_title("(a) ROC curve"); ax[0].legend(loc="lower right", title="AUC-ROC", title_fontsize=8.5)
ax[1].axhline(y_te.mean(), ls="--", c="k", lw=0.8, alpha=0.5)
ax[1].text(0.62, y_te.mean() + 0.012, f"prevalence {y_te.mean()*100:.2f}%", fontsize=7.6, style="italic")
ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
ax[1].set_title("(b) Precision-Recall curve"); ax[1].legend(loc="upper right", title="AUC-PR", title_fontsize=8.5)
fig.tight_layout(pad=0.6)
save(fig, "fig_roc_pr.png")

# metric bars
rows = []
for m in order:
    p = MODELS[m]; yp = (p >= 0.5).astype(int)
    rows.append(dict(model=m, auc=roc_auc_score(y_te, p), aucpr=average_precision_score(y_te, p),
                     f1=f1_score(y_te, yp), prec=precision_score(y_te, yp),
                     rec=recall_score(y_te, yp), brier=brier_score_loss(y_te, p)))
dfm = pd.DataFrame(rows).set_index("model")
dfm.to_csv(OUT + "_table_model_perf.csv")
fig, ax = plt.subplots(figsize=(TW, 2.5))
mets = ["auc", "aucpr", "f1", "prec", "rec"]
labs = ["AUC-ROC", "AUC-PR", "F1", "Precision", "Recall"]
x = np.arange(len(mets)); w = 0.2
for i, m in enumerate(order):
    v = [dfm.loc[m, k] for k in mets]
    bb = ax.bar(x + (i - 1.5) * w, v, w, label=m, color=cols[m], edgecolor="black", linewidth=0.4)
    ax.bar_label(bb, fmt="%.2f", fontsize=6.4, padding=1.5)
ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel("Score"); ax.set_ylim(0, 1.02)
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
ax.set_title("Test-set performance at the default 0.5 cut-off", pad=18)
fig.tight_layout(pad=0.5)
save(fig, "fig_model_bars.png")

# ================================================== FIG: threshold / operating
tt = pd.read_csv(RERUN + "table_threshold_tuned.csv")
rm = pd.read_csv(RERUN + "table_recall_matched.csv")
fig, ax = plt.subplots(1, 2, figsize=(TW, 2.5))
f1o = tt[tt.setting == "@F1-optimal"].set_index("model")
f5 = tt[tt.setting == "@0.50 (published)"].set_index("model")
idx = np.arange(len(order)); w = 0.36
b1 = ax[0].bar(idx - w/2, [f5.loc[m, "f1"] for m in order], w, label="at 0.50",
               color="#c6c6c6", edgecolor="black", linewidth=0.4)
b2 = ax[0].bar(idx + w/2, [f1o.loc[m, "f1"] for m in order], w, label="F1-optimal",
               color=[cols[m] for m in order], edgecolor="black", linewidth=0.4)
ax[0].bar_label(b1, fmt="%.3f", fontsize=6.6, padding=1); ax[0].bar_label(b2, fmt="%.3f", fontsize=6.6, padding=1)
ax[0].set_xticks(idx); ax[0].set_xticklabels(order, rotation=18, ha="right")
ax[0].set_ylabel("F1 score"); ax[0].set_ylim(0, 0.60); ax[0].legend(loc="upper left")
ax[0].set_title("(a) Effect of tuning the cut-off")

rmi = rm.set_index("model")
b3 = ax[1].bar(idx, [rmi.loc[m, "precision"] for m in order], 0.55,
               color=[cols[m] for m in order], edgecolor="black", linewidth=0.4)
ax[1].bar_label(b3, fmt="%.3f", fontsize=7.2, padding=2)
ax[1].set_xticks(idx); ax[1].set_xticklabels(order, rotation=18, ha="right")
ax[1].set_ylabel("Precision at recall = 0.420"); ax[1].set_ylim(0, 0.68)
ax[1].set_title("(b) Precision at matched recall")
fig.tight_layout(pad=0.6)
save(fig, "fig_threshold.png")

# ============================================================ FIG: calibration
import xgboost as xgb
bst = xgb.Booster(); bst.load_model(APP + "model.json")
d = xgb.DMatrix(X_te[FEAT].astype(float).values, feature_names=FEAT)
p_raw = bst.predict(d)
cal = json.load(open(APP + "calibration.json"))
p_cal = np.interp(p_raw, cal["x"], cal["y"])

def reliability(y, p, bins=10, min_n=10):
    """Equal-width bins, dropping any bin with too few patients to mean
    anything. At 6.3% prevalence the highest bins can hold one or two
    patients, and plotting those makes the curve look like noise."""
    e = np.linspace(0, 1, bins + 1)
    xs, ys, ns = [], [], []
    for i in range(bins):
        m = (p >= e[i]) & (p <= e[i + 1] if i == bins - 1 else p < e[i + 1])
        if m.sum() >= min_n:
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(m.sum())
    return np.array(xs), np.array(ys), np.array(ns)

fig, ax = plt.subplots(1, 2, figsize=(TW, 2.55))
ax[0].plot([0, 1], [0, 1], "k--", lw=0.9, label="perfect calibration")
for p, lab, col in [(p_raw, "Uncalibrated", C["acc"]), (p_cal, "Isotonic-calibrated", C["ok"])]:
    xs, ys, ns = reliability(y_te, p)
    ax[0].plot(xs, ys, "-", lw=1.4, color=col,
               label=f"{lab} (Brier {brier_score_loss(y_te,p):.4f})")
    # Marker area tracks how many patients back each point.
    ax[0].scatter(xs, ys, s=14 + 90 * np.sqrt(ns / ns.max()), color=col,
                  edgecolor="black", linewidth=0.5, zorder=3)
ax[0].set_xlabel("Mean predicted risk"); ax[0].set_ylabel("Observed frequency")
ax[0].set_title("(a) Reliability diagram"); ax[0].legend(loc="upper left")
ax[0].text(0.97, 0.06, "marker size $\propto$ patients in bin", transform=ax[0].transAxes,
           ha="right", fontsize=7.2, style="italic", color="#444444")
ax[0].set_xlim(-0.02, 1.02); ax[0].set_ylim(-0.02, 1.02)

ax[1].hist(p_raw, bins=40, alpha=0.6, color=C["acc"], label=f"Uncalibrated (mean {p_raw.mean()*100:.1f}%)")
ax[1].hist(p_cal, bins=40, alpha=0.6, color=C["ok"], label=f"Calibrated (mean {p_cal.mean()*100:.1f}%)")
ax[1].axvline(y_te.mean(), ls="--", c="k", lw=1.0)
ax[1].text(y_te.mean() + 0.02, ax[1].get_ylim()[1] * 0.72,
           f"true prevalence\n{y_te.mean()*100:.2f}%", fontsize=7.6, style="italic")
ax[1].set_xlabel("Predicted risk"); ax[1].set_ylabel("Patients"); ax[1].set_yscale("log")
ax[1].set_title("(b) Predicted-risk distribution"); ax[1].legend(loc="upper right")
fig.tight_layout(pad=0.6)
save(fig, "fig_calibration.png")

# ======================================================== FIG: operating points
ops = meta["operating_points"]
fig, ax = plt.subplots(1, 2, figsize=(TW, 2.5))
pr, rc, th = precision_recall_curve(y_te, p_cal)
ax[0].plot(rc, pr, lw=1.5, color=C["xgb"])
mk = {"screening": "o", "balanced": "s", "confirmatory": "^"}
for name, o in ops.items():
    t = o["test"]
    ax[0].plot(t["recall"], t["precision"], mk[name], ms=8, mec="black", mew=0.7,
               label=f"{name} (thr {o['threshold']:.3f})")
ax[0].axhline(y_te.mean(), ls="--", c="k", lw=0.8, alpha=0.5)
ax[0].set_xlabel("Recall"); ax[0].set_ylabel("Precision")
ax[0].set_title("(a) Operating points on the PR curve"); ax[0].legend(loc="upper right")

names = list(ops); xx = np.arange(len(names)); w = 0.26
for i, (k, lab, col) in enumerate([("precision", "Precision", C["xgb"]),
                                   ("recall", "Recall", C["ok"]),
                                   ("flag_rate", "Flag rate", "#c6c6c6")]):
    vals = [ops[n]["test"][k] for n in names]
    bb = ax[1].bar(xx + (i - 1) * w, vals, w, label=lab, color=col, edgecolor="black", linewidth=0.4)
    ax[1].bar_label(bb, fmt="%.2f", fontsize=6.6, padding=1.5)
ax[1].set_xticks(xx); ax[1].set_xticklabels([n.capitalize() for n in names])
ax[1].set_ylabel("Value"); ax[1].set_ylim(0, 1.0)
ax[1].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.17))
ax[1].set_title("(b) Held-out behaviour of each setting", pad=17)
fig.tight_layout(pad=0.6)
save(fig, "fig_operating_points.png")
print("batch 1 done")
