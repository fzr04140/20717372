# ======================================
# Selective Fusion by Percentile Thresholds
# ======================================
import pandas as pd
import numpy as np

print("=== Step 1: Coverage-based Selective Fusion ===")

df = pd.read_csv("data/norm_results_with_n.csv")

bbn_col = "Probability"
llm_col = "llm_probability_norm"
n_col = "n_samples"

parent_cols = [c for c in df.columns if c.startswith("Parent_")]

#Percentage threshold to try
percent_list = [0,10,20,30,40,50,60,70,80,90,100]

percentile_values = {}
for p in percent_list:
    percentile_values[p] = np.percentile(df[n_col], p)

print("\nComputed percentile thresholds:")
for p in percent_list:
    print(f"P{p}: n_threshold = {percentile_values[p]:.2f}")

# Generate CPT with different percentages one by one
for p in percent_list:
    threshold = percentile_values[p]
    df_p = df.copy()

    # selective fusion
    df_p["p_fused_raw"] = np.where(
        df_p[n_col] <= threshold,
        df_p[llm_col],      # use LLM
        df_p[bbn_col],      # use BBN
    )

    # Internal normalization for each node/parent combination
    df_p["p_fused_norm"] = np.nan

    for node, df_node in df_p.groupby("Target_Node"):
        used_parents = [c for c in parent_cols if df_node[c].notna().any()]

        if not used_parents:
            df_p.loc[df_node.index, "p_fused_norm"] = df_node["p_fused_raw"] / df_node["p_fused_raw"].sum()
        else:
            grouped = df_node.groupby(used_parents)["p_fused_raw"].transform(
                lambda x: x / x.sum() if x.sum() > 0 else x
            )
            df_p.loc[df_node.index, "p_fused_norm"] = grouped

    out_csv = f"data/norm_results_fused_P{p}.csv"
    df_p.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_csv}")

print("\n=== Step 1 Done ===")
