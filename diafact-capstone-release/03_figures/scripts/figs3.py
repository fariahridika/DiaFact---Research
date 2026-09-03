"""Batch 3: system and method diagrams, laid out to fit the 6.2in text width.
Note: matplotlib is not LaTeX -- underscores are literal, so no escaping."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle

OUT = r"D:/thesis paper/Capstone_Project_Template_for_Final_Book/figures/"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
TW = 6.2
BLUE, GREEN, GREY, RED, AMBER = "#2166ac", "#1a7f5a", "#5a5a5a", "#b2182b", "#b8860b"


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fc="white", ec=BLUE, fs=8.2, bold=False, lw=1.1, r=2.2, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3,
                fontweight="bold" if bold else "normal", linespacing=1.5, color=tc)


def arrow(ax, p, q, color=GREY, lw=1.2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, linewidth=lw,
                                 color=color, zorder=1, linestyle=ls, shrinkA=1.5, shrinkB=1.5))


def save(fig, n):
    fig.savefig(OUT + n); plt.close(fig); print("  saved", n)


# ============================================================ research pipeline
fig, ax = canvas(TW, 2.5)
stages = [
    ("1. Data\nPreparation", "5,437 patients\nWinsorisation\n6 derived features"),
    ("2. Feature\nSelection", "Boruta\n20 to 16 features"),
    ("3. Model\nTraining", "XGBoost, LightGBM\nCatBoost, AutoGluon\nOptuna, 5-fold CV"),
    ("4. Explanation", "SHAP TreeExplainer\nLIME cross-check"),
    ("5. Prescription", "SHAP-guided DiCE\nAlignment Score"),
]
w, gap = 17.2, 3.5
x0 = (100 - (len(stages) * w + (len(stages) - 1) * gap)) / 2
for i, (t, sub) in enumerate(stages):
    x = x0 + i * (w + gap)
    col = GREEN if i >= 3 else BLUE
    box(ax, x, 44, w, 28, t, fc="#eef4fa" if i < 3 else "#e9f5f0", ec=col, fs=8.4, bold=True)
    ax.text(x + w / 2, 39, sub, ha="center", va="top", fontsize=7.0, color="#333333", linespacing=1.55)
    if i:
        arrow(ax, (x - gap + 0.4, 58), (x - 0.6, 58))
ax.text(50, 93, "Research pipeline", ha="center", fontsize=10.5, fontweight="bold")
mid = x0 + 3 * (w + gap) - gap / 2
ax.plot([x0, mid], [82, 82], lw=1.0, color=BLUE)
ax.plot([mid, x0 + 5 * (w + gap) - gap], [82, 82], lw=1.0, color=GREEN)
ax.text((x0 + mid) / 2, 84.5, "Prediction stage", ha="center", fontsize=7.8, color=BLUE, style="italic")
ax.text((mid + x0 + 5 * (w + gap) - gap) / 2, 84.5, "Prescription stage", ha="center",
        fontsize=7.8, color=GREEN, style="italic")
save(fig, "fig_pipeline.png")

# ======================================================== system architecture
fig, ax = canvas(TW, 4.0)
ax.text(50, 97, "DiaFact three-tier architecture", ha="center", fontsize=10.5, fontweight="bold")


def band(y, h, label, tech, fc, ec):
    box(ax, 4, y, 92, h, "", fc=fc, ec=ec, lw=0.9, r=1.5)
    ax.text(6.5, y + h - 3.4, label, fontsize=8.0, style="italic", color=GREY, va="center")
    ax.text(93.5, y + h - 3.4, tech, fontsize=7.4, color=GREY, ha="right", va="center")


band(75, 18, "Presentation tier", "React 18 + Vite  ·  port 5173", "#f7f9fc", "#c9d6e4")
box(ax, 9, 77.5, 25, 10, "Assessment\nForm", fc="white", ec=BLUE, fs=8.0)
box(ax, 37.5, 77.5, 25, 10, "Results: risk,\nSHAP, plans", fc="white", ec=BLUE, fs=8.0)
box(ax, 66, 77.5, 25, 10, "History &\nVisit Compare", fc="white", ec=BLUE, fs=8.0)

band(45, 18, "Application tier", "Node.js + Express  ·  port 3001", "#f7fbf9", "#cfe3da")
box(ax, 9, 47.5, 25, 10, "Validation &\nsanitisation", fc="white", ec=GREEN, fs=8.0)
box(ax, 37.5, 47.5, 25, 10, "REST API\n/predict   /patients", fc="white", ec=GREEN, fs=7.8, bold=True)
box(ax, 66, 47.5, 25, 10, "Helmet, CORS,\nrate limit", fc="white", ec=GREEN, fs=8.0)

box(ax, 4, 8, 56, 26, "", fc="#fdf6f6", ec="#e8cdcd", lw=0.9, r=1.5)
ax.text(6.5, 30.6, "Inference tier", fontsize=8.0, style="italic", color=GREY, va="center")
ax.text(58, 30.6, "Flask  ·  5001", fontsize=7.4, color=GREY, ha="right", va="center")
box(ax, 7, 11.5, 16, 14, "XGBoost\n+ isotonic\ncalibration", fc="white", ec=RED, fs=7.6)
box(ax, 25.5, 11.5, 16, 14, "SHAP\nTreeExplainer", fc="white", ec=RED, fs=7.6)
box(ax, 44, 11.5, 13, 14, "Consistency-\nsafe DiCE", fc="white", ec=RED, fs=7.4)

box(ax, 63, 8, 33, 26, "", fc="#fdfaf2", ec="#e6d7ae", lw=0.9, r=1.5)
ax.text(65.5, 30.6, "Persistence tier", fontsize=8.0, style="italic", color=GREY, va="center")
ax.text(93.5, 30.6, "MySQL · 3306", fontsize=7.0, color=GREY, ha="right", va="center")
box(ax, 67, 11.5, 25, 14, "patients\n\npatient_visits", fc="white", ec=AMBER, fs=8.0)

arrow(ax, (47, 75), (47, 63.2), lw=1.3, style="<|-|>")
ax.text(48.5, 69, "HTTPS / JSON", fontsize=7.2, color=GREY, va="center")
arrow(ax, (30, 45), (30, 34.4), lw=1.3, style="<|-|>")
ax.text(31.5, 39.7, "JSON", fontsize=7.2, color=GREY, va="center")
arrow(ax, (79, 45), (79, 34.4), lw=1.3, style="<|-|>")
ax.text(80.5, 39.7, "SQL", fontsize=7.2, color=GREY, va="center")
save(fig, "fig_architecture.png")

# ================================================= consistency-safe generation
fig, ax = canvas(TW, 3.5)
ax.text(50, 96, "Why the search ranges over raw variables only", ha="center",
        fontsize=10.5, fontweight="bold")

ax.text(25, 87, "(a) Unconstrained search", ha="center", fontsize=9.0, fontweight="bold", color=RED)
box(ax, 3, 62, 44, 17, "DiCE varies all 16 features,\nincluding bmi, bp_ratio, bmi_age",
    fc="#fdf0f0", ec=RED, fs=7.8)
arrow(ax, (25, 62), (25, 52), color=RED)
box(ax, 3, 33, 44, 18, "weight 54 to 52 kg  (down)\nBMI 22.5 to 27.2  (up)\nheight unchanged",
    fc="white", ec=RED, fs=7.8)
arrow(ax, (25, 33), (25, 23), color=RED)
box(ax, 5, 9, 40, 13, "Physically impossible\nin 95-100% of cases", fc="#fdf0f0", ec=RED,
    fs=8.2, bold=True, tc=RED)

ax.plot([50, 50], [8, 84], color="#cccccc", lw=0.9, ls="--")

ax.text(75, 87, "(b) Consistency-safe search", ha="center", fontsize=9.0, fontweight="bold", color=GREEN)
box(ax, 53, 62, 44, 17, "DiCE varies only 5 raw variables:\nglucose, weight, SBP, DBP, pulse",
    fc="#eef7f3", ec=GREEN, fs=7.8)
arrow(ax, (75, 62), (75, 52), color=GREEN)
box(ax, 53, 33, 44, 18, "Derived features recomputed\nfrom the proposal before\nevery prediction",
    fc="white", ec=GREEN, fs=7.8)
arrow(ax, (75, 33), (75, 23), color=GREEN)
box(ax, 55, 9, 40, 13, "Inconsistency impossible\nby construction", fc="#eef7f3", ec=GREEN,
    fs=8.2, bold=True, tc=GREEN)
save(fig, "fig_consistency.png")

# ============================================================== ER diagram
fig, ax = canvas(TW, 3.0)
ax.text(50, 96, "Database schema", ha="center", fontsize=10.5, fontweight="bold")


def table(ax, x, y, w, title, rows, ec):
    hh, rh = 8.0, 6.2
    ax.add_patch(Rectangle((x, y + len(rows) * rh), w, hh, facecolor=ec,
                           edgecolor="black", linewidth=0.9, zorder=2))
    ax.text(x + w / 2, y + len(rows) * rh + hh / 2, title, ha="center", va="center",
            fontsize=8.6, fontweight="bold", color="white", zorder=3)
    for i, (nm, ty) in enumerate(reversed(rows)):
        yy = y + i * rh
        ax.add_patch(Rectangle((x, yy), w, rh, facecolor="white", edgecolor="#bbbbbb",
                               linewidth=0.6, zorder=2))
        ax.text(x + 2.2, yy + rh / 2, nm, ha="left", va="center", fontsize=7.0, zorder=3)
        ax.text(x + w - 2.2, yy + rh / 2, ty, ha="right", va="center", fontsize=6.4,
                color="#666666", zorder=3)


table(ax, 4, 26, 38, "patients", [
    ("id  (PK)", "INT AI"), ("name", "VARCHAR(150)"), ("age", "INT"),
    ("gender", "VARCHAR(10)"), ("created_at", "DATETIME")], BLUE)
table(ax, 58, 8, 38, "patient_visits", [
    ("id  (PK)", "INT AI"), ("patient_id  (FK)", "INT"), ("visit_number", "INT"),
    ("input_features", "JSON"), ("model_features", "JSON"), ("risk_score", "FLOAT"),
    ("risk_label", "VARCHAR(20)"), ("shap_values", "JSON"),
    ("counterfactuals", "JSON"), ("visited_at", "DATETIME")], GREEN)
arrow(ax, (42, 45), (58, 48), color=GREY, lw=1.2)
ax.text(50, 52, "1 : N", ha="center", fontsize=8.0, style="italic")
ax.text(50, 40, "ON DELETE\nCASCADE", ha="center", fontsize=6.8, color=GREY, linespacing=1.4)
ax.text(77, 2.5, "UNIQUE (patient_id, visit_number)", ha="center", fontsize=6.8,
        style="italic", color=GREY)
save(fig, "fig_er.png")

# ============================================================== use case
fig, ax = canvas(TW, 3.1)
ax.text(50, 97, "Use case diagram", ha="center", fontsize=10.5, fontweight="bold")
ax.add_patch(Rectangle((26, 5), 52, 84, facecolor="none", edgecolor="#999999", lw=1.0))
ax.text(52, 85.5, "DiaFact system", ha="center", fontsize=8.2, style="italic", color=GREY)


def actor(ax, x, y, label):
    ax.add_patch(Ellipse((x, y + 9), 5.0, 6.5, facecolor="white", edgecolor="black", lw=1.0))
    ax.plot([x, x], [y + 5.5, y - 2], color="black", lw=1.0)
    ax.plot([x - 4, x + 4], [y + 3, y + 3], color="black", lw=1.0)
    ax.plot([x, x - 3.5], [y - 2, y - 9], color="black", lw=1.0)
    ax.plot([x, x + 3.5], [y - 2, y - 9], color="black", lw=1.0)
    ax.text(x, y - 13, label, ha="center", fontsize=8.0, fontweight="bold")


actor(ax, 11, 48, "Clinician")
actor(ax, 90, 48, "ML Service")

ucs = [("Register / look up patient", 77), ("Enter clinical measurements", 64.5),
       ("View calibrated risk", 52), ("Inspect SHAP explanation", 39.5),
       ("Review action plans", 27), ("Compare with previous visit", 14.5)]
for t, y in ucs:
    ax.add_patch(Ellipse((52, y), 45, 10.0, facecolor="#eef4fa", edgecolor=BLUE, lw=1.0, zorder=2))
    ax.text(52, y, t, ha="center", va="center", fontsize=7.8, zorder=3)
    arrow(ax, (15.5, 46), (30.0, y), color="#888888", lw=0.8, style="-")
for t, y in ucs[2:5]:
    arrow(ax, (85.5, 46), (74.0, y), color="#888888", lw=0.8, style="-")
save(fig, "fig_usecase.png")
print("batch 3 done")
