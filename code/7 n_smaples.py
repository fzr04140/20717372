import pandas as pd

# === Real 2024 data ===
df_data = pd.read_csv("data/genie_ready_2024_rebinned.csv")

# === LLM norm file ===
df = pd.read_csv("data/norm_results_gpt.csv", encoding="utf-8-sig")  # Use "utf-8-sig" or "latin1" if errors occur

# === Find all Parent_ columns ===
parent_cols = [c for c in df.columns if c.startswith("Parent_")]
print("Parent Node Column:", parent_cols)

# =============================
# 1) State → Actual value mapping
# =============================
state_to_value = {
    "weather_conditions": {f"State{i}": i for i in range(1, 10)},
    "light_conditions": {"State1": 1, "State4": 4, "State5": 5, "State6": 6, "State7": 7},
    "road_surface_conditions": {"State1": 1, "State2": 2, "State3": 3, "State4": 4, "State5": 5, "State9": 9},
    "road_type": {"State1": 1, "State2": 2, "State3": 3, "State6": 6, "State7": 7, "State9": 9},
    "urban_or_rural_area": {"State1": 1, "State2": 2, "State3": 3},
    "speed_limit": {
        "State20": 20, "State30": 30, "State40": 40,
        "State50": 50, "State60": 60, "State70": 70
    },
    "sex_of_driver": {"State1": 1, "State2": 2, "State3": 3},
    "collision_severity": {"State1": 1, "State2": 2, "State3": 3},

    "age_band_of_driver": {
        "Young": "Young", "Adult": "Adult", "Senior": "Senior"
    },
    "vehicle_manoeuvre": {
        "GoingAhead": "GoingAhead", "Turning": "Turning", "Overtaking": "Overtaking",
        "AvoidingOrReversing": "AvoidingOrReversing", "ParkingRelated": "ParkingRelated",
        "PassengerPickDrop": "PassengerPickDrop"
    },
    "casualty_type": {
        "Pedestrian": "Pedestrian", "Cyclist": "Cyclist", "VehicleOccupant": "VehicleOccupant"
    },
    "number_of_vehicles": {
        "SingleVehicle": "SingleVehicle", "TwoVehicles": "TwoVehicles",
        "MultiVehicle": "MultiVehicle", "LargeCrash": "LargeCrash"
    },
    "number_of_casualties": {
        "One": "One", "Two": "Two", "Few": "Few", "Many": "Many"
    },
}

def map_state(var, state):
    """Map StateX / text in LLM to actual values in genie_ready_2024_rebinned.csv"""
    if pd.isna(state):
        return None
    mapping = state_to_value.get(var)
    if mapping is None:
        # If no mapping exists, use the original value
        return state
    return mapping.get(str(state), state)

# =============================
# 2) Calculate n_samples row by row
# =============================
n_list = []

for idx, row in df.iterrows():
    # Determine which parent nodes are active in this row (non-NaN)
    active_parents = [pc for pc in parent_cols if pd.notna(row[pc])]
    
    # No parent nodes = root node, define n as 0 (no concept of "parent combination frequency")
    if not active_parents:
        n_list.append(0)
        continue

    # Filter in real data: satisfy all parent variable conditions = current row's parent states
    cond = pd.Series(True, index=df_data.index)
    for pc in active_parents:
        var = pc.replace("Parent_", "")          # Column name in real data
        val = map_state(var, row[pc])           # Map to actual value (number / text)
        cond &= (df_data[var] == val)

    n = int(cond.sum())  # Number of occurrences of this parent combination in real 2024 data
    n_list.append(n)

# Write back to df
df["n_samples"] = n_list

# Save
df.to_csv("data/norm_results_with_n.csv", index=False, encoding="utf-8-sig")

print("\nSaved as norm_results_with_n.csv")
print("  Total rows:", len(df))
print("  Number of rows with non-zero samples:", (df["n_samples"] > 0).sum())