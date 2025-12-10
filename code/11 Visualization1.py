import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/fusion_results_metrics.csv")

plt.style.use("seaborn-v0_8-whitegrid")

# ========== fig 1：Macro-F1 & BA ==========
fig, ax = plt.subplots(figsize=(8,5))

color1 = "#1f77b4"
color2 = "#ff7f0e"

ax.plot(
    df["Percent"], df["Macro-F1"],
    marker='o', linewidth=1.8, color=color1, label="Macro-F1"
)
ax.plot(
    df["Percent"], df["BalancedAcc"],
    marker='o', linewidth=1.8, color=color2, label="Balanced Accuracy"
)

ax.set_xlabel("Fusion Percent (%)", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(fontsize=11)
ax.set_title("Macro-F1 and Balanced Accuracy vs Fusion Percent", fontsize=14)


ax.set_xticks(df["Percent"])


for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("data/fig_macro_balanced.png", dpi=300, bbox_inches="tight")
plt.close()


# ========== fig 2：Recall per class ==========
plt.figure(figsize=(8,5))

plt.plot(df["Percent"], df["Recall_Fatal"], marker='o', label="Class1")
plt.plot(df["Percent"], df["Recall_Serious"], marker='o', label="Class2")
plt.plot(df["Percent"], df["Recall_Slight"], marker='o', label="Class3")

plt.xlabel("Fusion Percent (p)", fontsize=12)
plt.ylabel("Recall", fontsize=12)
plt.title("Recall per Class vs Fusion Percent", fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.xticks(df["Percent"])

plt.savefig("data/fig_recall_all.png", dpi=300, bbox_inches="tight")
plt.close()


# ========== fig 3：n_sample vs Percentile ==========
df_n = pd.read_csv("data/norm_results_with_n.csv")

percent_list = [0,10,20,30,40,50,60,70,80,90,100]
n_threshold = [np.percentile(df_n["n_samples"], p) for p in percent_list]

plt.figure(figsize=(8,6))
plt.plot(percent_list, n_threshold, marker='o', linewidth=2)

plt.xlabel("Percentile p (%)", fontsize=12)
plt.ylabel("n_samples threshold", fontsize=12)
plt.title("Percentile Threshold vs n_samples", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)


for x, y in zip(percent_list, n_threshold):
    plt.text(x, y, f"{int(y)}", fontsize=9, ha='center', va='bottom', color='black')


plt.xticks(percent_list)

plt.savefig("data/fig_nsample.png", dpi=300, bbox_inches="tight")
plt.close()
