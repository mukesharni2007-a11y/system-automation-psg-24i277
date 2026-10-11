import pandas as pd
import random

# Number of rows
n_rows = 2000

data = []

for i in range(n_rows):

    # Generate sensor values
    temperature = round(random.uniform(50, 95), 2)
    pressure = round(random.uniform(4.5, 10.0), 2)
    vibration = round(random.uniform(0.3, 5.0), 2)
    rpm = random.randint(2500, 3200)
    current = round(random.uniform(7.0, 16.0), 2)

    # --------------------------------
    # SAME RULES AS start.py
    # --------------------------------

    # Fault
    if (
        temperature >= 90
        or pressure >= 9.0
        or vibration >= 4.0
        or rpm < 2600
        or rpm > 3150
        or current >= 15.0
    ):
        state = "Fault"

    # Warning
    elif (
        temperature >= 80
        or pressure >= 7.0
        or vibration >= 2.5
        or rpm < 2750
        or rpm > 3050
        or current >= 13.5
    ):
        state = "Warning"

    # Normal
    else:
        state = "Normal"

    # Action
    if state == "Fault":
        action = "EMERGENCY_STOP"
    elif state == "Warning":
        action = "REDUCE_LOAD"
    else:
        action = "CONTINUE_NORMAL"

    data.append([
        temperature,
        pressure,
        vibration,
        rpm,
        current,
        state,
        action
    ])


# --------------------------------
# CREATE DATAFRAME
# --------------------------------

df = pd.DataFrame(data, columns=[
    "Temperature",
    "Pressure",
    "Vibration",
    "RPM",
    "Current",
    "State",
    "Action"
])


# --------------------------------
# SAVE CSV
# --------------------------------

df.to_csv("sensor_data.csv", index=False)

print("CSV file created successfully!")
print("File: sensor_data.csv")
print(f"Total rows: {len(df)}")

print("\nState distribution:")
print(df["State"].value_counts())

print("\nFirst 10 rows:")
print(df.head(10))