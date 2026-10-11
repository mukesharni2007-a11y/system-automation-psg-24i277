import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------------------
# 1. Load dataset
# -----------------------------------------

df = pd.read_csv("generate_sensor_data.csv")

print("Dataset loaded successfully!")
print("Total rows:", len(df))

print("\nClass distribution:")
print(df["State"].value_counts())


# -----------------------------------------
# 2. Features and target
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
# 3. Train-test split
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------------------
# 4. Random Forest
# -----------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42
)


# -----------------------------------------
# 5. Train
# -----------------------------------------

model.fit(X_train, y_train)

print("\nRandom Forest trained successfully!")


# -----------------------------------------
# 6. Predict
# -----------------------------------------

y_pred = model.predict(X_test)


# -----------------------------------------
# 7. Accuracy
# -----------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"Accuracy: {accuracy * 100:.2f}%")


# -----------------------------------------
# 8. Classification report
# -----------------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -----------------------------------------
# 9. Confusion matrix
# -----------------------------------------

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# -----------------------------------------
# 10. Feature importance
# -----------------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)


# -----------------------------------------
# 11. Save model
# -----------------------------------------

joblib.dump(model, "random_forest_model.pkl")

print("\nModel saved as random_forest_model.pkl")