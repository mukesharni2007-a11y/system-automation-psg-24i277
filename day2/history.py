import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------
# LOAD COMMON CSV FILE
# --------------------------------

df = pd.read_csv("../sensor_data.csv")

print("Dataset loaded successfully!")
print(f"Total rows: {len(df)}")


# --------------------------------
# FEATURES
# --------------------------------

X = df[
    [
        "Temperature",
        "Pressure",
        "Vibration",
        "RPM",
        "Current"
    ]
]

# Target
y = df["State"]


# --------------------------------
# SPLIT DATA
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------
# RANDOM FOREST MODEL
# --------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    max_features=None
)


# --------------------------------
# TRAIN
# --------------------------------

model.fit(X_train, y_train)

print("\nModel training completed!")


# --------------------------------
# PREDICTION
# --------------------------------

y_pred = model.predict(X_test)


# --------------------------------
# ACCURACY
# --------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print("       MODEL PERFORMANCE")
print("================================")

print(f"Accuracy: {accuracy * 100:.2f}%")


# --------------------------------
# CLASSIFICATION REPORT
# --------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --------------------------------
# SAVE MODEL
# --------------------------------

joblib.dump(model, "random_forest_model.pkl")

print("\n================================")
print("Model saved successfully!")
print("Location: day2/random_forest_model.pkl")
print("================================")