"""Project Gantt chart, April 2025 to July 2026."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from datetime import date

OUT = r"D:/thesis paper/Capstone_Project_Template_for_Final_Book/figures/"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9.0, "figure.dpi": 200, "savefig.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

BLUE, GREEN, AMBER, RED, PURPLE = "#2166ac", "#1a7f5a", "#b8860b", "#b2182b", "#6a51a3"

# (task, start, end, colour, group)
TASKS = [
    ("Problem definition and scoping",        date(2025, 4, 1),  date(2025, 5, 31), BLUE),
    ("Literature review",                     date(2025, 4, 15), date(2025, 8, 15), BLUE),
    ("Dataset acquisition and cleaning",      date(2025, 6, 1),  date(2025, 7, 31), GREEN),
    ("Feature engineering and Boruta",        date(2025, 7, 1),  date(2025, 8, 31), GREEN),
    ("Model training and tuning",             date(2025, 8, 1),  date(2025, 10, 15), GREEN),
    ("Ablation study",                        date(2025, 9, 15), date(2025, 11, 15), GREEN),
    ("SHAP and LIME analysis",                date(2025, 10, 1), date(2025, 12, 15), AMBER),
    ("Counterfactual generation (v1)",        date(2025, 11, 1), date(2026, 1, 15), AMBER),
    ("Metric re-examination and correction",  date(2026, 1, 1),  date(2026, 2, 28), RED),
    ("Consistency-safe regeneration",         date(2026, 2, 1),  date(2026, 3, 31), RED),
    ("Web application development",           date(2026, 1, 15), date(2026, 4, 30), PURPLE),
    ("Calibration and safety constraints",    date(2026, 3, 15), date(2026, 5, 15), PURPLE),
    ("Security hardening and testing",        date(2026, 4, 1),  date(2026, 5, 31), PURPLE),
    ("Deployed-system evaluation",            date(2026, 5, 1),  date(2026, 6, 15), PURPLE),
    ("Book writing and documentation",        date(2026, 4, 15), date(2026, 7, 15), BLUE),
    ("Final review and submission",           date(2026, 6, 15), date(2026, 7, 31), BLUE),
]

MILESTONES = [
    ("Baselines\ncomplete",      date(2025, 10, 15)),
    ("Metric flaw\nidentified",  date(2026, 1, 10)),
    ("Application\ndeployed",    date(2026, 4, 30)),
    ("Submission",               date(2026, 7, 31)),
]

fig, ax = plt.subplots(figsize=(6.2, 4.3))
ypos = range(len(TASKS))
for i, (name, s, e, col) in enumerate(TASKS):
    y = len(TASKS) - 1 - i
    ax.barh(y, (e - s).days, left=s, height=0.62, color=col, alpha=0.85,
            edgecolor="black", linewidth=0.45, zorder=3)

ax.set_yticks([len(TASKS) - 1 - i for i in range(len(TASKS))])
ax.set_yticklabels([t[0] for t in TASKS], fontsize=8.0)
ax.set_ylim(-2.6, len(TASKS) - 0.3)

ax.set_xlim(date(2025, 3, 20), date(2026, 8, 10))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
plt.setp(ax.get_xticklabels(), fontsize=6.6)
ax.grid(axis="x", alpha=0.25, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

# Milestones sit on their own row under the bars, labelled ABOVE the marker so
# the text cannot collide with the month axis underneath.
ax.axhline(-1.55, color="#cccccc", lw=0.7, zorder=1)
for name, d in MILESTONES:
    ax.plot(d, -1.55, "D", ms=5.0, color="black", zorder=5)
    ax.annotate(name, xy=(d, -1.55), xytext=(0, 8), textcoords="offset points",
                ha="center", va="bottom", fontsize=6.4, style="italic",
                color="#333333", linespacing=1.25)

handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.4, label=l) for c, l in
           [(BLUE, "Scoping & writing"), (GREEN, "Data & modelling"),
            (AMBER, "Explainability"), (RED, "Correction phase"),
            (PURPLE, "Application & deployment")]]
ax.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13),
          fontsize=7.4, frameon=False)
ax.set_title("Project schedule, April 2025 to July 2026", fontsize=10.2,
             fontweight="bold", pad=26)
fig.tight_layout(pad=0.5)
fig.savefig(OUT + "fig_gantt.png")
print("  saved fig_gantt.png")
