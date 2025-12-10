import pandas as pd

# ======================
# Step 0: Load data
# ======================

df = pd.read_csv("data/genie_ready_2024.csv")
print("Loaded dataset:", df.shape)

# ======================
# Step 1: age_band_of_driver → 3 bins
# ======================

def encode_age_band(x):
    x = int(x)
    if x <= 3:
        return "Young"
    elif x <= 7:
        return "Adult"
    else:
        return "Senior"

df["age_band_of_driver"] = df["age_band_of_driver"].astype(int).apply(encode_age_band)


# ======================
# Step 2: casualty_type → 3 bins
# ======================

def encode_casualty_type(x):
    x = int(x)
    if x in [0, 1, 2, 3]:  # pedestrians
        return "Pedestrian"
    elif x in [8, 9, 10, 11, 16, 17, 18, 19]:  # cyclists & motorcyclists
        return "Cyclist"
    else:
        return "VehicleOccupant"

df["casualty_type"] = df["casualty_type"].astype(int).apply(encode_casualty_type)


# ======================
# Step 3: vehicle_manoeuvre → 5 bins
# ======================

def encode_vehicle_manoeuvre(x):
    x = int(x)

    if x in [1, 2, 3]:
        return "GoingAhead"          # Going straight ahead/starting/decelerating

    elif x in [4, 5, 6]:
        return "Turning"             # Turning left/right, U-turn

    elif x in [7, 8, 9]:
        return "Overtaking"          # Overtaking, changing lanes, merging

    elif x in [10, 11, 12]:
        return "ParkingRelated"      # Parking/starting related

    elif x in [13, 14]:
        return "PassengerPickDrop"   # Passenger pick-up/drop-off

    else:  # x in [15,16,17,18,19,20,...]
        return "AvoidingOrReversing" # Avoiding obstacles, reversing, other abnormal maneuvers


df["vehicle_manoeuvre"] = df["vehicle_manoeuvre"].astype(int).apply(encode_vehicle_manoeuvre)


# ======================
# Step 4: number_of_vehicles → 4 bins
# ======================

def encode_num_vehicles(x):
    x = int(x)
    if x == 1:
        return "SingleVehicle"
    elif x == 2:
        return "TwoVehicles"
    elif x <= 4:
        return "MultiVehicle"
    else:
        return "LargeCrash"

df["number_of_vehicles"] = df["number_of_vehicles"].astype(int).apply(encode_num_vehicles)


# ======================
# Step 5: number_of_casualties → 4 bins
# ======================

def encode_num_casualties(x):
    x = int(x)
    if x == 1:
        return "One"
    elif x == 2:
        return "Two"
    elif x <= 5:
        return "Few"
    else:
        return "Many"

df["number_of_casualties"] = df["number_of_casualties"].astype(int).apply(encode_num_casualties)


# ======================
# Step 6: Output results
# ======================

df = df.astype(str)  
output_path = "data/genie_ready_2024_rebinned.csv"
df.to_csv(output_path, index=False)

print("Saved rebinned dataset to:", output_path)
print("New unique value counts:")
print(df.nunique())
