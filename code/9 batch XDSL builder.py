# ======================================
# Step 2：Batch update XDSL for all percentiles (Final Correct Version)
# ======================================
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from itertools import product
import os

print("\n=== Step 2: batch XDSL builder (Final) ===")

xdsl_template = "genie_ready_2024.xdsl"

def get_node_states(cpt_node):
    return [s.get("id") for s in cpt_node.findall("state")]

percent_list = [0,10,20,30,40,50,60,70,80,90,100]

for p in percent_list:

    cpt_csv = f"data/norm_results_fused_P{p}.csv"
    out_xdsl = f"genie_fused_P{p}.xdsl"

    df = pd.read_csv(cpt_csv, encoding="utf-8-sig")
    df.columns = df.columns.str.replace('\ufeff', '', regex=False)

    # ---- Load XDSL ----
    tree = ET.parse(xdsl_template)
    root = tree.getroot()

    # ---- Read node states ----
    node_states = {}
    for cpt in root.findall(".//cpt"):
        nid = cpt.get("id")             # e.g., road_type
        node_states[nid] = get_node_states(cpt)

    # ---- Update CPT ----
    for cpt in root.findall(".//cpt"):
        node_id = cpt.get("id")         # variable name

        if node_id not in df["Target_Node"].unique():
            continue

        df_node = df[df["Target_Node"] == node_id]
        states = node_states[node_id]

        parents_elem = cpt.find("parents")
        probs_flat = []

        # ===== No parent =====
        if parents_elem is None or not parents_elem.text.strip():
            for s in states:
                row = df_node[df_node["Target_State"] == s]
                val = float(row["p_fused_norm"].values[0]) if not row.empty else 1/len(states)
                probs_flat.append(val)

        # ===== Has parents =====
        else:
            parents = parents_elem.text.strip().split()     # e.g., ['road_type', 'speed_limit']
            parent_state_lists = [node_states[p] for p in parents]
            combos = list(product(*parent_state_lists))

            for combo in combos:

                subset = df_node.copy()
                for i, p in enumerate(parents):
                    subset = subset[subset[f"Parent_{p}"] == combo[i]]

                for s in states:
                    row = subset[subset["Target_State"] == s]
                    val = float(row["p_fused_norm"].values[0]) if not row.empty else 1/len(states)
                    probs_flat.append(val)

        # ---- Write probabilities back ----
        prob_elem = cpt.find("probabilities")
        prob_elem.text = " ".join(f"{v:.10f}" for v in probs_flat)

    tree.write(out_xdsl, encoding="utf-8", xml_declaration=True)
    print(f"Saved XDSL: {out_xdsl}")

print("\n=== Step 2 Done ===")
