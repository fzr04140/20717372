import pandas as pd
from collections import defaultdict

# === Step 1: Load Data ===

cpt_df = pd.read_csv("cpt_extracted.csv")
code_df = pd.read_excel("dft-road-casualty-statistics-road-safety-open-dataset-data-guide-2024.xlsx",
                         sheet_name="2024_code_list")

# === Step 2: Official codebook (for unbucketed variables) ===

code_clean = code_df.dropna(subset=["field name", "code/format", "label"])
official_codebook = defaultdict(dict)

for _, row in code_clean.iterrows():
    field = row["field name"].strip()
    code = str(row["code/format"]).strip()
    label = row["label"].strip()
    official_codebook[field][code] = label

# === Step 3: Customize the rebinned codebook (used for the variables after bucketing)

rebinned_codebook = {
    "age_band_of_driver": {
        "Young": "young driver (< 30 years)",
        "Adult": "adult driver (30–60 years)",
        "Senior": "senior driver (> 60 years)"
    },
    "vehicle_manoeuvre": {
        "GoingAhead": "going ahead / straight driving",
        "Turning": "turning at junction",
        "Overtaking": "overtaking or changing lanes",
        "AvoidingOrReversing": "avoiding / reversing maneuver",
        "ParkingRelated": "parking related movement",
        "PassengerPickDrop": "picking up or dropping passenger"
    },
    "casualty_type": {
        "Pedestrian": "pedestrian",
        "Cyclist": "cyclist",
        "VehicleOccupant": "vehicle occupant"
    },
    "number_of_vehicles": {
        "SingleVehicle": "single-vehicle crash",
        "TwoVehicles": "two-vehicle crash",
        "MultiVehicle": "multi-vehicle crash",
        "LargeCrash": "large scale crash (> 5 vehicles)"
    },
    "number_of_casualties": {
        "One": "one casualty",
        "Two": "two casualties",
        "Few": "few casualties (3–4)",
        "Many": "many casualties (≥ 5)"
    }
}

# === Step 4: Smart label parsing function ===

def get_label(var, value):

    # if  StateX → Extract the numbers
    if isinstance(value, str) and value.startswith("State"):
        raw_code = value[5:]
    else:
        raw_code = str(value)

    # see if it's rebinned first
    if var in rebinned_codebook:
        if raw_code in rebinned_codebook[var]:
            return rebinned_codebook[var][raw_code]
        else:
            return raw_code  # fallback

    # Check the official codebook.

    if raw_code in official_codebook.get(var, {}):
        return official_codebook[var][raw_code]

    # fallback
    return raw_code


# === Step 5: generate Prompt ===
def extract_prompt(row):
    target_node = row['Target_Node']
    target_state = row['Target_State']

    target_label = get_label(target_node, target_state)
    target_node_clean = target_node.replace("_", " ").lower()

    # Concatenating parent conditions
    conditions = []
    for col in row.index:
        if col.startswith("Parent_") and pd.notna(row[col]):
            var = col.replace("Parent_", "")
            val = row[col]

            label = get_label(var, val)
            conditions.append(f"{var.replace('_', ' ').lower()} is {label}")

    if len(conditions) > 0:
        condition_text = "Given that " + ", ".join(conditions)
    else:
        condition_text = "In general"

    prompt = f"{condition_text}, what is the probability that {target_node_clean} is {target_label}?"
    return prompt


cpt_df["prompt"] = cpt_df.apply(extract_prompt, axis=1)
cpt_df.to_csv("data/llm_prompts.csv", index=False)

print("Saved: llm_prompts.csv")
