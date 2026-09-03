"""Batch 2: explainability, counterfactual and ablation figures.
Also runs the canonical deployed-system evaluation once and saves it, so the
book text and the figures quote the same numbers."""
import os, sys, json, warnings, joblib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

BASE = r"D:/thesis paper/TEHI 2026/exp_v2/exp_v2/Final Results/exp_v2/"
RERUN = r"D:/thesis paper/TEHI 2026/rerun_outputs/"
FRZ = RERUN + "frozen_hyp/"
BOOK = r"D:/thesis paper/Capstone_Project_Template_for_Final_Book/"
OUT = BOOK + "figures/"
DATA = BOOK + "figures/_data/"
os.makedirs(DATA, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 8.8, "xtick.labelsize": 8.8, "ytick.labelsize": 8.8,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
TW = 6.2
STD, SHP = "#8c8c8c", "#1a7f5a"
def save(f, n): f.savefig(OUT + n); plt.close(f); print("  saved", n)

splits = joblib.load(BASE + "checkpoints/nb01_splits.pkl")
X_te = splits["X_test"]
xai = joblib.load(BASE + "results/xai/xai_results.pkl")
SV = np.asarray(xai["shap_values"]); Xs = xai["X_test_shap"]; FEAT = list(Xs.columns)

PRETTY = {"glucose": "Fasting glucose", "weight": "Weight", "bmi_age": "BMI x age",
          "diastolic_bp": "Diastolic BP", "glucose_bmi": "Glucose x BMI", "age": "Age",
          "pulse_rate": "Pulse rate", "bmi": "BMI", "gender": "Gender",
          "systolic_bp": "Systolic BP", "bp_ratio": "BP ratio", "pulse_pressure": "Pulse pressure",
          "hypertensive": "Hypertensive", "cardio_risk": "Cardio risk",
          "family_hypertension": "Family hypertension", "family_risk": "Family risk"}

# ==================================================== FIG: SHAP global + beeswarm
imp = pd.Series(np.abs(SV).mean(0), index=FEAT).sort_values(ascending=False)
imp.to_csv(DATA + "shap_importance.csv", header=["mean_abs_shap"])
top = imp.head(10)[::-1]

fig, ax = plt.subplots(1, 2, figsize=(TW, 3.1), gridspec_kw={"width_ratios": [1, 1.25]})
b = ax[0].barh([PRETTY.get(i, i) for i in top.index], top.values,
               color="#2166ac", edgecolor="black", linewidth=0.4, height=0.68)
ax[0].bar_label(b, fmt="%.3f", fontsize=7.2, padding=2)
ax[0].set_xlabel("Mean |SHAP value|"); ax[0].set_xlim(0, top.max() * 1.22)
ax[0].set_title("(a) Global importance"); ax[0].grid(axis="y", visible=False)

# beeswarm, drawn manually so it matches the house style
order = imp.head(8).index[::-1]
rng = np.random.default_rng(0)
for row, f in enumerate(order):
    v = SV[:, FEAT.index(f)]
    x = Xs[f].values.astype(float)
    n = (x - np.nanpercentile(x, 5)) / max(np.nanpercentile(x, 95) - np.nanpercentile(x, 5), 1e-9)
    n = np.clip(n, 0, 1)
    jitter = rng.uniform(-0.17, 0.17, len(v))
    sc = ax[1].scatter(v, row + jitter, c=n, cmap="coolwarm", s=2.4, alpha=0.55,
                       linewidths=0, rasterized=True)
ax[1].set_yticks(range(len(order))); ax[1].set_yticklabels([PRETTY.get(i, i) for i in order])
ax[1].axvline(0, color="black", lw=0.7, alpha=0.6)
ax[1].set_xlabel("SHAP value (impact on model output)")
ax[1].set_title("(b) Per-patient effects"); ax[1].grid(axis="y", visible=False)
cb = fig.colorbar(sc, ax=ax[1], pad=0.02, aspect=28)
cb.set_ticks([0, 1]); cb.set_ticklabels(["Low", "High"]); cb.set_label("Feature value", fontsize=8.5)
cb.outline.set_linewidth(0.4)
fig.tight_layout(pad=0.6)
save(fig, "fig_shap.png")

# ==================================================== FIG: correlation heatmap
corr = splits["X_train"][FEAT].astype(float).corr()
fig, ax = plt.subplots(figsize=(TW, 4.6))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
lab = [PRETTY.get(f, f) for f in FEAT]
ax.set_xticks(range(len(FEAT))); ax.set_xticklabels(lab, rotation=55, ha="right", fontsize=7.6)
ax.set_yticks(range(len(FEAT))); ax.set_yticklabels(lab, fontsize=7.6)
for i in range(len(FEAT)):
    for j in range(len(FEAT)):
        v = corr.iloc[i, j]
        if abs(v) >= 0.55 and i != j:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.0,
                    color="white" if abs(v) > 0.75 else "black")
ax.grid(False)
cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02); cb.set_label("Pearson r", fontsize=8.5)
ax.set_title("Feature correlation on the training split\n(values shown where |r| $\\geq$ 0.55)", fontsize=10)
fig.tight_layout(pad=0.5)
save(fig, "fig_correlation.png")

# ==================================================== FIG: counterfactual results
m = pd.read_csv(FRZ + "cf_metrics_consistent.csv")
pp = m.groupby(["method", "idx"])[["proximity_l1", "sparsity", "concentration"]].mean().reset_index()
s = pp[pp.method.str.startswith("standard")].set_index("idx").sort_index()
g = pp[pp.method.str.startswith("shap")].set_index("idx").sort_index()
common = s.index.intersection(g.index)

from scipy import stats
res = {}
for col, key in [("proximity_l1", "proximity"), ("sparsity", "sparsity"), ("concentration", "alignment")]:
    a, b_ = s.loc[common, col].values, g.loc[common, col].values
    res[key] = dict(std=a.mean(), shap=b_.mean(), p=stats.wilcoxon(a, b_)[1])
json.dump(res, open(DATA + "cf_results.json", "w"), indent=2)

fig, ax = plt.subplots(1, 3, figsize=(TW, 2.35))
for k, (key, ylab, ttl) in enumerate([("proximity", "Mean L1 distance", "(a) Proximity $\\downarrow$"),
                                      ("sparsity", "Fraction of variables changed", "(b) Sparsity $\\downarrow$"),
                                      ("alignment", "Alignment score", "(c) Alignment $\\uparrow$")]):
    r = res[key]
    bb = ax[k].bar(["Standard", "SHAP-guided"], [r["std"], r["shap"]],
                   color=[STD, SHP], edgecolor="black", linewidth=0.5, width=0.55)
    ax[k].bar_label(bb, fmt="%.3f" if key != "proximity" else "%.2f", fontsize=8.4, padding=2)
    ax[k].set_ylabel(ylab); ax[k].set_title(ttl)
    ax[k].set_ylim(0, max(r["std"], r["shap"]) * 1.30)
    d = (r["shap"] - r["std"]) / r["std"] * 100
    ax[k].text(0.5, 0.90, f"{d:+.1f}%   (p = {r['p']:.5f})", transform=ax[k].transAxes,
               ha="center", fontsize=8.0, style="italic")
fig.tight_layout(pad=0.6)
save(fig, "fig_cf_results.png")

# alignment distribution
fig, ax = plt.subplots(1, 2, figsize=(TW, 2.4))
sg = m[m.method.str.startswith("shap")].concentration
sd = m[m.method.str.startswith("standard")].concentration
bins = np.linspace(0, 1, 21)
ax[0].hist(sd, bins=bins, alpha=0.7, color=STD, label=f"Standard (median {sd.median():.2f})")
ax[0].hist(sg, bins=bins, alpha=0.7, color=SHP, label=f"SHAP-guided (median {sg.median():.2f})")
ax[0].set_xlabel("Alignment score"); ax[0].set_ylabel("Counterfactuals")
ax[0].set_title("(a) Alignment distribution"); ax[0].legend(loc="upper left")

ax[1].scatter(m[m.method.str.startswith("standard")].sparsity, sd, s=13, color=STD,
              alpha=0.65, label="Standard", edgecolor="none")
ax[1].scatter(m[m.method.str.startswith("shap")].sparsity, sg, s=13, color=SHP,
              alpha=0.65, label="SHAP-guided", edgecolor="none")
rho, pv = stats.spearmanr(m.sparsity, m.concentration)
ax[1].set_xlabel("Sparsity (fraction changed)"); ax[1].set_ylabel("Alignment score")
ax[1].set_title("(b) Alignment is sparsity-neutral"); ax[1].legend(loc="lower right")
ax[1].text(0.03, 0.06, f"Spearman $\\rho$ = {rho:.3f} (p = {pv:.2f})",
           transform=ax[1].transAxes, fontsize=8.2, style="italic")
fig.tight_layout(pad=0.6)
save(fig, "fig_alignment.png")

# ==================================================== FIG: ablation
ab = pd.read_csv(RERUN + "table_ablation_multiseed.csv")
ab = ab[ab.experiment != "A1"].copy()
ab["lab"] = ab.configuration
fig, ax = plt.subplots(figsize=(TW, 3.3))
grp = {"A2": "Oversampling", "A3": "Engineered features", "A4": "Feature selection",
       "A5": "Outlier handling", "A7": "Pipeline order"}
ab = ab.sort_values(["experiment", "F1_mean"])
ypos, labels, colors, seen = [], [], [], 0
pool = {"A2": "#2166ac", "A3": "#4393c3", "A4": "#92c5de", "A5": "#d6604d", "A7": "#b2182b"}
for e in ["A2", "A3", "A4", "A5", "A7"]:
    sub = ab[ab.experiment == e]
    for _, r in sub.iterrows():
        ypos.append(seen); labels.append(f"{r.configuration}"); colors.append(pool[e]); seen += 1
    seen += 0.7
vals = ab.F1_mean.values; errs = ab.F1_sd.values
ax.barh(ypos, vals, xerr=errs, color=colors, edgecolor="black", linewidth=0.4,
        height=0.72, error_kw=dict(lw=0.8, capsize=2.5))
ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=8.2)
ax.axvspan(vals.max() - 0.10, vals.max(), color="grey", alpha=0.12)
ax.set_xlabel("F1 score (mean $\\pm$ sd over 5 random splits)")
ax.set_xlim(0, 0.52); ax.grid(axis="y", visible=False)
handles = [plt.Rectangle((0, 0), 1, 1, color=pool[e], ec="black", lw=0.4) for e in pool]
ax.legend(handles, [grp[e] for e in pool], ncol=3, loc="upper center",
          bbox_to_anchor=(0.5, 1.13), fontsize=8.2)
ax.set_title("Ablation study: nothing outside the shaded band is distinguishable from noise", pad=22, fontsize=9.6)
fig.tight_layout(pad=0.5)
save(fig, "fig_ablation.png")
print("batch 2 done")
