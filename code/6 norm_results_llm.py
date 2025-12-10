import pandas as pd
from scipy.stats import entropy

# === Read data ===
df = pd.read_csv("data/llm_generated_gpt.csv", encoding="gbk")

# === Determine the grouping column ===
parent_cols = [col for col in df.columns if col.startswith("Parent_")]
group_cols = ["Target_Node"] + parent_cols

# Normalize LLM probabilities by group ===
def normalize_llm(group):
    probs = group["llm_probability"].fillna(0)
    total = probs.sum()
    if total > 0:
        group["llm_probability_norm"] = probs / total
    else:
        group["llm_probability_norm"] = 1.0 / len(group)
    return group

df = df.groupby(group_cols, group_keys=False, dropna=False).apply(normalize_llm)

# saved
df.to_csv("data/norm_results_gpt.csv", index=False)
print("saved successfully.")


