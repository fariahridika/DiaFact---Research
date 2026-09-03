"""Batch 4: deployed-system evaluation figure, from the saved canonical run."""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BOOK = r"D:/thesis paper/Capstone_Project_Template_for_Final_Book/"
OUT, DATA = BOOK + "figures/", BOOK + "figures/_data/"
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

df = pd.read_csv(DATA + "app_cf_records.csv")
S = json.load(open(DATA + "app_eval.json"))

fig, ax = plt.subplots(1, 3, figsize=(TW, 2.4))

# (a) number of changes
mx = int(df.n_changes.max())
bins = np.arange(0.5, mx + 1.5, 1)
for s, lab, col in [("standard", "Unconstrained", STD), ("shap_guided", "SHAP-guided", SHP)]:
    sub = df[df.strategy == s]
    ax[0].hist(sub.n_changes, bins=bins, alpha=0.72, color=col,
               label=f"{lab} (mean {sub.n_changes.mean():.2f})")
ax[0].set_xticks(range(1, mx + 1))
ax[0].set_xlabel("Variables changed per plan"); ax[0].set_ylabel("Action plans")
ax[0].set_title("(a) Plan length"); ax[0].legend(loc="upper right")

# (b) alignment
bb = np.linspace(0, 1, 21)
for s, lab, col in [("standard", "Unconstrained", STD), ("shap_guided", "SHAP-guided", SHP)]:
    sub = df[df.strategy == s]
    ax[1].hist(sub.alignment, bins=bb, alpha=0.72, color=col,
               label=f"{lab} (mean {sub.alignment.mean():.3f})")
ax[1].set_xlabel("SHAP alignment score"); ax[1].set_ylabel("Action plans")
ax[1].set_title("(b) Targeting quality"); ax[1].legend(loc="upper left")

# (c) direction violations: research vs deployed
labels = ["Unconstrained", "SHAP-guided"]
research = [75.0, 51.7]                      # measured in the research pipeline
deployed = [S["standard"]["direction_violations"] / max(S["standard"]["n_cfs"], 1) * 100,
            S["shap_guided"]["direction_violations"] / max(S["shap_guided"]["n_cfs"], 1) * 100]
x = np.arange(2); w = 0.36
b1 = ax[2].bar(x - w/2, research, w, label="Research pipeline", color="#d6604d",
               edgecolor="black", linewidth=0.4)
b2 = ax[2].bar(x + w/2, deployed, w, label="Deployed system", color=SHP,
               edgecolor="black", linewidth=0.4)
ax[2].bar_label(b1, fmt="%.0f%%", fontsize=7.6, padding=2)
ax[2].bar_label(b2, fmt="%.0f%%", fontsize=7.6, padding=2)
ax[2].set_xticks(x); ax[2].set_xticklabels(labels, fontsize=8.2)
ax[2].set_ylabel("Plans moving a variable\nthe wrong way (%)")
ax[2].set_ylim(0, 92); ax[2].legend(loc="upper right")
ax[2].set_title("(c) Clinically unsafe advice")
fig.tight_layout(pad=0.6)
fig.savefig(OUT + "fig_app_eval.png"); plt.close(fig)
print("  saved fig_app_eval.png")
