from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "historical_joint_readings.csv"
MODEL_PATH = BASE_DIR / "models" / "wear_model.joblib"
LOG_PATH = BASE_DIR / "data" / "maintenance_log.txt"

JOINTS = ["shoulder", "elbow", "wrist"]
FEATURE_COLUMNS = [
    "torque_nm", "temperature_c", "vibration_mms",
    "torque_change", "temperature_change", "vibration_change"
]

HARD_LIMITS = {
    "torque": 95.0,
    "temperature": 90.0,
    "vibration": 18.0
}

CAUTION_LIMITS = {
    "torque": 55.0,
    "temperature": 58.0,
    "vibration": 5.0
}

WEAR_LIMIT = 0.65
MONITOR_LIMIT = 0.35

NOTE_URGENT_TERMS = ["smoke", "burning smell", "crack", "joint jammed", "sparks", "fire"]
NOTE_WARNING_TERMS = [
    "grinding", "overheating", "excessive vibration", "loose", "bearing noise",
    "lubrication leak", "leak", "metal particles", "stiff", "abnormal torque", "wear"
]


def generate_dataset(path=DATA_PATH, rows=360, seed=42):
    rng = np.random.default_rng(seed)
    records = []
    for i in range(rows):
        wear = int(rng.random() < 0.30)
        if wear:
            t = float(np.clip(rng.normal(73.0, 10.0), 15.0, 94.0))
            temp = float(np.clip(rng.normal(73.0, 8.0), 25.0, 89.0))
            vib = float(np.clip(rng.normal(10.5, 2.5), 1.0, 17.5))
            dt = float(np.clip(abs(rng.normal(6.0, 1.2)), 0.5, 12.0))
            dtemp = float(np.clip(abs(rng.normal(4.0, 1.0)), 0.5, 8.0))
            dvib = float(np.clip(abs(rng.normal(2.0, 0.5)), 0.2, 4.0))
        else:
            t = float(np.clip(rng.normal(42.0, 7.0), 15.0, 94.0))
            temp = float(np.clip(rng.normal(45.0, 6.0), 25.0, 89.0))
            vib = float(np.clip(rng.normal(2.8, 1.0), 0.5, 17.5))
            dt = float(np.clip(abs(rng.normal(1.8, 0.6)), 0.1, 8.0))
            dtemp = float(np.clip(abs(rng.normal(0.9, 0.4)), 0.1, 5.0))
            dvib = float(np.clip(abs(rng.normal(0.4, 0.2)), 0.05, 2.0))

        records.append({
            "reading_id": i + 1,
            "torque_nm": round(t, 2),
            "temperature_c": round(temp, 2),
            "vibration_mms": round(vib, 2),
            "torque_change": round(dt, 2),
            "temperature_change": round(dtemp, 2),
            "vibration_change": round(dvib, 2),
            "wear_label": wear
        })

    df = pd.DataFrame(records)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def train_or_load_model(retrain=False):
    if retrain or not DATA_PATH.exists():
        generate_dataset(DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df["wear_label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    if not retrain and MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            class_weight="balanced"
        )
        model.fit(X_train, y_train)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

    preds = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "feature_importances": dict(zip(FEATURE_COLUMNS, [float(round(v, 4)) for v in model.feature_importances_])),
        "rows": len(df),
        "test_rows": len(y_test)
    }
    return model, metrics


def _parse_val(val):
    if val is None:
        return None
    try:
        num = float(val)
        if np.isnan(num) or np.isinf(num):
            return None
        return num
    except (ValueError, TypeError):
        return None


def detect_sensor_spoof(arm, previous_readings=None):
    reasons = []
    suspected = False

    if isinstance(previous_readings, dict):
        for j in JOINTS:
            if isinstance(arm.get(j), dict) and isinstance(previous_readings.get(j), dict):
                curr_t = _parse_val(arm[j].get("torque"))
                prev_t = _parse_val(previous_readings[j].get("torque"))
                if curr_t is not None and prev_t is not None:
                    delta = abs(curr_t - prev_t)
                    if delta > 35.0:
                        suspected = True
                        reasons.append(
                            f"{j.title()} torque shifted abruptly by {delta:.1f} Nm in a single cycle (>35 Nm limit)."
                        )

    for j in JOINTS:
        if isinstance(arm.get(j), dict):
            t = _parse_val(arm[j].get("torque"))
            temp = _parse_val(arm[j].get("temperature"))
            vib = _parse_val(arm[j].get("vibration"))
            if t is not None and temp is not None and vib is not None:
                if t > 80.0 and vib < 0.5 and temp < 35.0:
                    suspected = True
                    reasons.append(
                        f"{j.title()} reports contradictory state: heavy torque ({t:.1f} Nm) with near-zero vibration ({vib:.2f} mm/s) and cold temperature ({temp:.1f} °C)."
                    )
                if vib < 0.0 or temp < -40.0:
                    suspected = True
                    reasons.append(f"{j.title()} has physically impossible negative sensor readings.")

    return suspected, reasons


def analyze_maintenance_note(note):
    text = (note or "").lower()
    urgent = [w for w in NOTE_URGENT_TERMS if w in text]
    warnings = [w for w in NOTE_WARNING_TERMS if w in text]

    if urgent:
        return "URGENT", f"Urgent maintenance note keyword detected: {', '.join(urgent)}."
    if warnings:
        return "WARNING", f"Warning maintenance note keyword detected: {', '.join(warnings)}."
    return "NORMAL", "Routine note; no safety-critical keywords detected."


def load_maintenance_logs(path=LOG_PATH):
    if not Path(path).exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",", 2)]
            if len(parts) == 3:
                date_str, arm_id, text = parts
            else:
                date_str, arm_id, text = "-", "Arm A", line
            sig, exp = analyze_maintenance_note(text)
            records.append({
                "entry_id": idx,
                "date": date_str,
                "arm_id": arm_id,
                "note": text,
                "signal": sig,
                "explanation": exp
            })
    return records


def evaluate_arm(arm, model=None, maintenance_note="", previous_readings=None):
    reasons = []

    if not isinstance(arm, dict):
        return {
            "decision": "SCHEDULE_MAINTENANCE",
            "wear_probability": 0.0,
            "joint_wear_probabilities": {},
            "note_signal": "NORMAL",
            "spoof_suspected": True,
            "safety_stop_triggered": False,
            "reasons": ["Invalid sensor payload: expected a dictionary of joint readings."]
        }

    dangerous = []
    invalid_data = []

    for j in JOINTS:
        data = arm.get(j)
        if not isinstance(data, dict):
            invalid_data.append(f"Missing data block for {j} joint.")
            continue

        t = _parse_val(data.get("torque"))
        temp = _parse_val(data.get("temperature"))
        vib = _parse_val(data.get("vibration"))

        if t is None or temp is None or vib is None:
            invalid_data.append(f"Invalid or missing numeric reading in {j} joint.")
            continue

        if t >= HARD_LIMITS["torque"]:
            dangerous.append(f"{j.title()} torque {t:.1f} Nm exceeded hard limit ({HARD_LIMITS['torque']} Nm).")
        if temp >= HARD_LIMITS["temperature"]:
            dangerous.append(f"{j.title()} temperature {temp:.1f} °C exceeded hard limit ({HARD_LIMITS['temperature']} °C).")
        if vib >= HARD_LIMITS["vibration"]:
            dangerous.append(f"{j.title()} vibration {vib:.1f} mm/s exceeded hard limit ({HARD_LIMITS['vibration']} mm/s).")

    if dangerous:
        return {
            "decision": "EMERGENCY_STOP",
            "wear_probability": 1.0,
            "joint_wear_probabilities": {},
            "note_signal": analyze_maintenance_note(maintenance_note)[0],
            "spoof_suspected": False,
            "safety_stop_triggered": True,
            "reasons": ["Deterministic hard safety stop triggered independently of machine learning model."] + dangerous
        }

    spoof_detected, spoof_reasons = detect_sensor_spoof(arm, previous_readings)
    note_sig, note_exp = analyze_maintenance_note(maintenance_note)

    if invalid_data:
        return {
            "decision": "SCHEDULE_MAINTENANCE",
            "wear_probability": 0.0,
            "joint_wear_probabilities": {},
            "note_signal": note_sig,
            "spoof_suspected": True,
            "safety_stop_triggered": False,
            "reasons": ["Sensor integrity failure: inspect wiring and sensor calibration."] + invalid_data + spoof_reasons
        }

    joint_probs = {}
    if model is not None:
        for j in JOINTS:
            d = arm[j]
            feat_df = pd.DataFrame([{
                "torque_nm": float(d["torque"]),
                "temperature_c": float(d["temperature"]),
                "vibration_mms": float(d["vibration"]),
                "torque_change": 3.0,
                "temperature_change": 1.5,
                "vibration_change": 0.8
            }], columns=FEATURE_COLUMNS)
            joint_probs[j] = float(model.predict_proba(feat_df)[0][1])
        wear_prob = max(joint_probs.values()) if joint_probs else 0.0
    else:
        wear_prob = 0.0

    if spoof_detected:
        decision = "SCHEDULE_MAINTENANCE"
        reasons.append("Sensor spoofing or communication fault suspected.")
        reasons.extend(spoof_reasons)
    elif note_sig == "URGENT":
        decision = "SCHEDULE_MAINTENANCE"
        reasons.append(note_exp)
    elif wear_prob >= WEAR_LIMIT or note_sig == "WARNING":
        decision = "SCHEDULE_MAINTENANCE"
        if wear_prob >= WEAR_LIMIT:
            reasons.append(f"Predicted wear risk is high ({wear_prob:.1%}, threshold {WEAR_LIMIT:.0%}).")
        if note_sig == "WARNING":
            reasons.append(note_exp)
    elif wear_prob >= MONITOR_LIMIT or any(
        arm[j]["torque"] >= CAUTION_LIMITS["torque"] or
        arm[j]["temperature"] >= CAUTION_LIMITS["temperature"] or
        arm[j]["vibration"] >= CAUTION_LIMITS["vibration"]
        for j in JOINTS
    ):
        decision = "MONITOR"
        reasons.append(f"Advisory wear signal: wear probability is {wear_prob:.1%} or a joint reading reached caution levels.")
    else:
        decision = "SAFE"
        reasons.append(f"All joint metrics within normal limits. Predicted wear risk is {wear_prob:.1%}.")

    reasons.append("Hard limit safety checks verified clear before ML inference.")

    return {
        "decision": decision,
        "wear_probability": wear_prob,
        "joint_wear_probabilities": joint_probs,
        "note_signal": note_sig,
        "spoof_suspected": spoof_detected,
        "safety_stop_triggered": False,
        "reasons": reasons
    }


def make_demo_scenarios():
    return {
        "SAFE": {
            "shoulder": {"torque": 38.0, "temperature": 42.0, "vibration": 2.0},
            "elbow": {"torque": 42.0, "temperature": 45.0, "vibration": 2.8},
            "wrist": {"torque": 35.0, "temperature": 40.0, "vibration": 2.2},
            "maintenance_note": "Routine inspection completed; no unusual sound."
        },
        "MONITOR": {
            "shoulder": {"torque": 58.0, "temperature": 60.0, "vibration": 5.0},
            "elbow": {"torque": 54.0, "temperature": 56.0, "vibration": 4.5},
            "wrist": {"torque": 48.0, "temperature": 50.0, "vibration": 3.5},
            "maintenance_note": "Routine inspection completed."
        },
        "SCHEDULE_MAINTENANCE": {
            "shoulder": {"torque": 73.0, "temperature": 72.0, "vibration": 10.0},
            "elbow": {"torque": 68.0, "temperature": 69.0, "vibration": 9.0},
            "wrist": {"torque": 64.0, "temperature": 65.0, "vibration": 8.0},
            "maintenance_note": "Technician heard grinding and excessive vibration near elbow."
        },
        "EMERGENCY_STOP": {
            "shoulder": {"torque": 40.0, "temperature": 43.0, "vibration": 2.0},
            "elbow": {"torque": 45.0, "temperature": 94.0, "vibration": 3.0},
            "wrist": {"torque": 35.0, "temperature": 40.0, "vibration": 2.0},
            "maintenance_note": "Routine inspection."
        },
        "SPOOF": {
            "shoulder": {"torque": 88.0, "temperature": 29.0, "vibration": 0.2},
            "elbow": {"torque": 42.0, "temperature": 45.0, "vibration": 2.8},
            "wrist": {"torque": 35.0, "temperature": 40.0, "vibration": 2.2},
            "maintenance_note": "Routine inspection completed.",
            "previous_readings": {
                "shoulder": {"torque": 42.0, "temperature": 41.0, "vibration": 2.5},
                "elbow": {"torque": 41.0, "temperature": 43.0, "vibration": 2.6},
                "wrist": {"torque": 35.0, "temperature": 39.0, "vibration": 2.0}
            }
        }
    }


if __name__ == "__main__":
    df = generate_dataset()
    model, metrics = train_or_load_model(retrain=True)
    print(f"Generated {len(df)} samples. Test accuracy: {metrics['accuracy']:.1%}")
    for name, scn in make_demo_scenarios().items():
        res = evaluate_arm(scn, model, scn.get("maintenance_note", ""), scn.get("previous_readings"))
        print(f"  {name:22s} -> {res['decision']} (P={res['wear_probability']:.1%}, Spoof={res['spoof_suspected']})")
