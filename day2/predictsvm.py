import joblib
import pandas as pd


# Load model and scaler
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")


# Get sensor values
temperature = float(input("Enter Temperature: "))
pressure = float(input("Enter Pressure: "))
vibration = float(input("Enter Vibration: "))
rpm = float(input("Enter RPM: "))
current = float(input("Enter Current: "))


# Create input
input_data = pd.DataFrame({
    "Temperature": [temperature],
    "Pressure": [pressure],
    "Vibration": [vibration],
    "RPM": [rpm],
    "Current": [current]
})


# Scale input
input_scaled = scaler.transform(input_data)


# Predict
prediction = model.predict(input_scaled)[0]

probability = model.predict_proba(input_scaled)[0]
confidence = max(probability) * 100


# Display
print("\n-----------------------------")
print(" MACHINE PREDICTION")
print("-----------------------------")

print("Temperature :", temperature)
print("Pressure    :", pressure)
print("Vibration   :", vibration)
print("RPM         :", rpm)
print("Current     :", current)

print("\nPredicted State:", prediction)
print(f"Confidence   : {confidence:.2f}%")

print("-----------------------------")