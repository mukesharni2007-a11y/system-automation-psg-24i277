import joblib
import pandas as pd

# --------------------------------
# LOAD TRAINED MODEL
# --------------------------------

model = joblib.load("random_forest_model.pkl")

print("Random Forest model loaded successfully!")


# --------------------------------
# GET SENSOR INPUT
# --------------------------------

temperature = float(input("Enter Temperature: "))
pressure = float(input("Enter Pressure: "))
vibration = float(input("Enter Vibration: "))
rpm = float(input("Enter RPM: "))
current = float(input("Enter Current: "))


# --------------------------------
# CREATE INPUT DATA
# --------------------------------

input_data = pd.DataFrame({
    "Temperature": [temperature],
    "Pressure": [pressure],
    "Vibration": [vibration],
    "RPM": [rpm],
    "Current": [current]
})


# --------------------------------
# PREDICT
# --------------------------------

prediction = model.predict(input_data)[0]

probabilities = model.predict_proba(input_data)[0]

confidence = max(probabilities) * 100


# --------------------------------
# DISPLAY RESULT
# --------------------------------

print("\n======================================")
print("       SMART FACTORY AI PREDICTION")
print("======================================")

print(f"Temperature : {temperature} °C")
print(f"Pressure    : {pressure} bar")
print(f"Vibration   : {vibration}")
print(f"RPM         : {rpm}")
print(f"Current     : {current} A")

print("--------------------------------------")

print(f"Predicted State : {prediction}")
print(f"Confidence      : {confidence:.2f}%")

print("======================================")