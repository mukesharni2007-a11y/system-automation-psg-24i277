import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------------------
# 1. Load dataset
# -----------------------------------------

df = pd.read_csv("generate_sensor_data.csv")

print("Dataset loaded successfully!")
print("Total rows:", len(df))


# -----------------------------------------
# 2. Input features and target
# -----------------------------------------

X = df[
    [
        "Temperature",
        "Pressure",
        "Vibration",
        "RPM",
        "Current"
    ]
]

y = df["State"]


# -----------------------------------------
# 3. Split dataset
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# -----------------------------------------
# 4. Scale the features
# -----------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------------------
# 5. Create SVM model
# -----------------------------------------

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True,
    random_state=42
)


# -----------------------------------------
# 6. Train model
# -----------------------------------------

model.fit(X_train_scaled, y_train)

print("\nSVM model trained successfully!")


# -----------------------------------------
# 7. Prediction
# -----------------------------------------

y_pred = model.predict(X_test_scaled)


# -----------------------------------------
# 8. Evaluation
# -----------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# -----------------------------------------
# 9. Save model and scaler
# -----------------------------------------

joblib.dump(model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nSVM model saved as svm_model.pkl")
print("Scaler saved as scaler.pkl")