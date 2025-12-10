# ======================================
# Step 3：Auto Evaluation for α Fusion (Final Robust Version)
# ======================================
import pysmile
import pysmile_license
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score

print("\n=== Step 3: Auto Evaluation (Fusion Strength Search) ===")

# ==============================
# Load Test Dataset
# ==============================
df_test = pd.read_csv("data/genie_ready_2023_rebinned.csv")  

def normalize_key(x):
    return str(x).strip()


# ======================================
# Universal Mapping Table (for matching)
# ======================================
value_to_state = {
    "collision_severity": {
        "1":"State1", "2":"State2", "3":"State3"
    },

    "weather_conditions": {str(k):f"State{k}" for k in range(1,10)},

    "light_conditions": {
        "1":"State1", "4":"State4", "5":"State5",
        "6":"State6", "7":"State7"
    },

    "road_surface_conditions": {
        "1":"State1","2":"State2","3":"State3",
        "4":"State4","5":"State5","9":"State9"
    },

    "road_type": {
        "1":"State1","2":"State2","3":"State3",
        "6":"State6","7":"State7","9":"State9"
    },

    "urban_or_rural_area": {
        "1":"State1","2":"State2","3":"State3"
    },

    "sex_of_driver": {
        "1":"State1","2":"State2","3":"State3"
    },

    "speed_limit": {
        "20":"State20","30":"State30", "40":"State40",
        "50":"State50","60":"State60","70":"State70"
    },

    "age_band_of_driver": {
        "Young":"Young", "Adult":"Adult", "Senior":"Senior"
    },

    "vehicle_manoeuvre": {
        "GoingAhead":"GoingAhead",
        "Turning":"Turning",
        "Overtaking":"Overtaking",
        "AvoidingOrReversing":"AvoidingOrReversing",
        "ParkingRelated":"ParkingRelated",
        "PassengerPickDrop":"PassengerPickDrop"
    },

    "casualty_type": {
        "Pedestrian":"Pedestrian",
        "Cyclist":"Cyclist",
        "VehicleOccupant":"VehicleOccupant"
    },

    "number_of_vehicles": {
        "SingleVehicle":"SingleVehicle",
        "TwoVehicles":"TwoVehicles",
        "MultiVehicle":"MultiVehicle",
        "LargeCrash":"LargeCrash"
    },

    "number_of_casualties": {
        "One":"One","Two":"Two","Few":"Few","Many":"Many"
    }
}


# ======================================
# Percent List (Fusion Strength α)
# ======================================
percent_list = [0,10,20,30,40,50,60,70,80,90,100]
results = []

# ==========================================================
# Helper：safe evidence setter 
# ==========================================================
def safe_set_evidence(net, col, val):
    """returns True if valid, False if invalid"""
    vkey = normalize_key(val)

    if vkey not in value_to_state[col]:
        return False

    mapped = value_to_state[col][vkey]

    # Real state list of XDSL
    allowed_states = net.get_outcome_ids(col)

    if mapped not in allowed_states:
        return False

    try:
        net.set_evidence(col, mapped)
    except:
        return False

    return True


# ==========================================================
# (1) Baseline: P0 =  BBN
# ==========================================================
print("\n Evaluating Baseline BBN (P0)...")

net_base = pysmile.Network()
net_base.read_file("genie_fused_P0.xdsl")

y_true_base, y_pred_base = [], []
skip_base = 0

for idx, row in tqdm(df_test.iterrows(), total=len(df_test)):

    net_base.clear_all_evidence()
    valid = True

    for col, val in row.items():
        if col == "collision_severity":
            continue
        if not safe_set_evidence(net_base, col, val):
            valid = False
            break

    if not valid:
        skip_base += 1
        continue

    net_base.update_beliefs()
    probs = net_base.get_node_value("collision_severity")
    y_pred_base.append(np.argmax(probs)+1)
    y_true_base.append(row["collision_severity"])

print(f"Baseline skipped samples: {skip_base}")

# ---- Baseline Metrics ----
macroF_base = f1_score(y_true_base, y_pred_base, average="macro")
weightedF_base = f1_score(y_true_base, y_pred_base, average="weighted")

rec1_b = recall_score(y_true_base, y_pred_base, labels=[1], average="macro")
rec2_b = recall_score(y_true_base, y_pred_base, labels=[2], average="macro")
rec3_b = recall_score(y_true_base, y_pred_base, labels=[3], average="macro")

BA_base = (rec1_b + rec2_b + rec3_b) / 3

print(f"Baseline  Macro-F1={macroF_base:.4f}, BalancedAcc={BA_base:.4f}")


# ==========================================================
# (2) Loop Over All Percent (Fusion Strength)
# ==========================================================
for p in percent_list:

    xdsl = f"genie_fused_P{p}.xdsl"
    print(f"\n--- Testing P{p} ({xdsl}) ---")

    net = pysmile.Network()
    net.read_file(xdsl)

    y_true, y_pred = [], []
    skip_cnt = 0

    for idx, row in tqdm(df_test.iterrows(), total=len(df_test)):
        net.clear_all_evidence()

        valid = True
        for col, val in row.items():
            if col == "collision_severity":
                continue
            if not safe_set_evidence(net, col, val):
                valid = False
                break

        if not valid:
            skip_cnt += 1
            continue

        net.update_beliefs()
        probs = net.get_node_value("collision_severity")

        y_pred.append(np.argmax(probs)+1)
        y_true.append(row["collision_severity"])

    print(f"Skipped samples (P{p}): {skip_cnt}")

    # ---- Metrics ----
    macroF = f1_score(y_true, y_pred, average="macro")
    weightedF = f1_score(y_true, y_pred, average="weighted")

    rec1 = recall_score(y_true, y_pred, labels=[1], average="macro")
    rec2 = recall_score(y_true, y_pred, labels=[2], average="macro")
    rec3 = recall_score(y_true, y_pred, labels=[3], average="macro")

    BA = (rec1 + rec2 + rec3) / 3

    # ---- Improvement ----
    imp_macro = (macroF - macroF_base) / macroF_base * 100
    imp_wF = (weightedF - weightedF_base) / weightedF_base * 100
    imp_r1 = (rec1 - rec1_b) / rec1_b * 100 if rec1_b>0 else np.nan
    imp_r2 = (rec2 - rec2_b) / rec2_b * 100
    imp_BA = (BA - BA_base) / BA_base * 100

    results.append([
        p, macroF, weightedF, BA,
        rec1, rec2, rec3,
        imp_macro, imp_wF, imp_r1, imp_r2, imp_BA
    ])

# ==========================================================
# Save Results
# ==========================================================
df_out = pd.DataFrame(
    results,
    columns=[
        "Percent","Macro-F1","Weighted-F1","BalancedAcc",
        "Recall_Fatal","Recall_Serious","Recall_Slight",
        "Improve_MacroF(%)","Improve_WeightedF(%)",
        "Improve_Recall1(%)","Improve_Recall2(%)","Improve_BA(%)"
    ]
)

df_out.to_csv("data/fusion_results_metrics.csv", index=False, encoding="utf-8-sig")
print("\nSaved  fusion_results_metrics.csv")
print("\n=== Step 3 Done ===")
