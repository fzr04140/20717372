import pandas as pd

# Step 1: Load raw data
collision = pd.read_csv("data/dft-road-casualty-statistics-collision-2024.csv")
vehicle   = pd.read_csv("data/dft-road-casualty-statistics-vehicle-2024.csv")
casualty  = pd.read_csv("data/dft-road-casualty-statistics-casualty-2024.csv")

# Step 2: Standardize primary key type
for df in [collision, vehicle, casualty]:
    df["collision_index"] = df["collision_index"].astype(str)

# Step 3: Select basic fields from collision (excluding vehicle/casualty counts)
collision_filtered = collision[[
    "collision_index",  
    "collision_severity", "weather_conditions", "light_conditions", "road_surface_conditions",
    "road_type", "speed_limit", "urban_or_rural_area"
]]

# Step 4: Extract key fields from vehicle and casualty (for merging)
vehicle_filtered = vehicle[["collision_index", "sex_of_driver", "age_band_of_driver", "vehicle_manoeuvre"]]
casualty_filtered = casualty[["collision_index", "casualty_type"]]

# Step 5: Count vehicles and casualties per accident
vehicle_counts = vehicle.groupby("collision_index").size().reset_index(name="number_of_vehicles")
casualty_counts = casualty.groupby("collision_index").size().reset_index(name="number_of_casualties")

# Step 6: Merge multiple tables (stepwise)
merged = collision_filtered.merge(vehicle_filtered, on="collision_index", how="inner")
merged = merged.merge(casualty_filtered, on="collision_index", how="inner")
merged = merged.merge(vehicle_counts, on="collision_index", how="left")
merged = merged.merge(casualty_counts, on="collision_index", how="left")

# Step 7: Handle missing values: replace "-1" strings and actual NA values
merged.replace("-1", pd.NA, inplace=True)  # string-type -1
merged.replace(-1, pd.NA, inplace=True)    # numeric-type -1 (for safety)
cleaned = merged.dropna()

# Step 8: Remove primary key
df = cleaned.drop(columns=["collision_index"])

# Step 9: Convert all to string and export in GeNIe required format
df = df.astype(str)
df.to_csv("data/genie_ready_2024.csv", index=False)

print("saved as genie_ready_2024.csv")
print(f"rows:{df.shape[0]}, columns:{df.shape[1]}")