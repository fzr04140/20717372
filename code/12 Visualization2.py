import pandas as pd
import matplotlib.pyplot as plt

# ===== 1. Read data =====
df = pd.read_csv("data/fusion_results_metrics.csv")

#Select the fusion ratio to be displayed
BEST_P = 90

row = df.loc[df["Percent"] == BEST_P].iloc[0]

# ===== 2. Get the improvement indicators that need to be displayed =====

metric_labels = [
    "Macro-F1",
    "Balanced Accuracy",
    "Recall_Fatal",
    "Recall_Serious"
]

values = [
    row["Improve_MacroF(%)"],
    row["Improve_BA(%)"],
    row["Improve_Recall1(%)"],
    row["Improve_Recall2(%)"]
]

# ===== 3.Draw a horizontal bar chart of improvement percentage  =====
plt.style.use("seaborn-v0_8")

fig, ax = plt.subplots(figsize=(8, 5))

colors = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]

y_pos = range(len(metric_labels))
bars = ax.barh(y_pos, values, color=colors)

ax.xaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

ax.set_yticks(y_pos)
ax.set_yticklabels(metric_labels, fontsize=11)

ax.set_xscale("log")
ax.set_xlabel("Improvement (%)  (log scale)", fontsize=12)

ax.set_title(f"Performance Improvement at Fusion Percent = {BEST_P}%", fontsize=14)

# Mark the specific value (xx.x%) at the end of each column
for bar, v in zip(bars, values):
    ax.text(
        bar.get_width() * 1.02,    
        bar.get_y() + bar.get_height() / 2,
        f"{v:.1f}%",
        va="center",
        fontsize=10
    )

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("data/improvement_p90.png", dpi=300, bbox_inches="tight")
