import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from health_monitor import (
    train_or_load_model,
    evaluate_arm,
    make_demo_scenarios,
    generate_dataset,
    analyze_maintenance_note,
    load_maintenance_logs,
    HARD_LIMITS,
    CAUTION_LIMITS,
    DATA_PATH
)

st.set_page_config(
    page_title="Robotic Arm Joint Health Monitor",
    page_icon="🤖",
    layout="wide"
)

model, metrics = train_or_load_model()
scenarios = make_demo_scenarios()

st.sidebar.title("Arm Controls")
scenario_options = [
    "Normal Operation (SAFE)",
    "Advisory Condition (MONITOR)",
    "Wear Risk & Warning Note (SCHEDULE_MAINTENANCE)",
    "High Temperature (EMERGENCY_STOP)",
    "Suspicious Torque Jump (SPOOF)",
    "Manual Slider Controls"
]
selected_mode = st.sidebar.selectbox("Scenario Preset", scenario_options)

active_scenario = None
if selected_mode == "Normal Operation (SAFE)":
    active_scenario = scenarios["SAFE"]
elif selected_mode == "Advisory Condition (MONITOR)":
    active_scenario = scenarios["MONITOR"]
elif selected_mode == "Wear Risk & Warning Note (SCHEDULE_MAINTENANCE)":
    active_scenario = scenarios["SCHEDULE_MAINTENANCE"]
elif selected_mode == "High Temperature (EMERGENCY_STOP)":
    active_scenario = scenarios["EMERGENCY_STOP"]
elif selected_mode == "Suspicious Torque Jump (SPOOF)":
    active_scenario = scenarios["SPOOF"]
else:
    st.sidebar.subheader("Shoulder Joint (J1)")
    s_t = st.sidebar.slider("Torque (Nm)", 10.0, 110.0, 42.0, 0.5, key="s_t")
    s_temp = st.sidebar.slider("Temperature (°C)", 20.0, 105.0, 45.0, 0.5, key="s_temp")
    s_vib = st.sidebar.slider("Vibration (mm/s)", 0.1, 22.0, 2.2, 0.1, key="s_vib")

    st.sidebar.subheader("Elbow Joint (J2)")
    e_t = st.sidebar.slider("Torque (Nm)", 10.0, 110.0, 45.0, 0.5, key="e_t")
    e_temp = st.sidebar.slider("Temperature (°C)", 20.0, 105.0, 48.0, 0.5, key="e_temp")
    e_vib = st.sidebar.slider("Vibration (mm/s)", 0.1, 22.0, 3.0, 0.1, key="e_vib")

    st.sidebar.subheader("Wrist Joint (J3)")
    w_t = st.sidebar.slider("Torque (Nm)", 10.0, 110.0, 35.0, 0.5, key="w_t")
    w_temp = st.sidebar.slider("Temperature (°C)", 20.0, 105.0, 40.0, 0.5, key="w_temp")
    w_vib = st.sidebar.slider("Vibration (mm/s)", 0.1, 22.0, 1.8, 0.1, key="w_vib")

    custom_note = st.sidebar.text_input("Attached Technician Note", "Routine inspection completed.")
    test_jump = st.sidebar.checkbox("Simulate previous reading jump", value=False)
    prev = None
    if test_jump:
        prev = {
            "shoulder": {"torque": s_t - 42.0},
            "elbow": {"torque": e_t},
            "wrist": {"torque": w_t}
        }

    active_scenario = {
        "shoulder": {"torque": s_t, "temperature": s_temp, "vibration": s_vib},
        "elbow": {"torque": e_t, "temperature": e_temp, "vibration": e_vib},
        "wrist": {"torque": w_t, "temperature": w_temp, "vibration": w_vib},
        "maintenance_note": custom_note,
        "previous_readings": prev
    }

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset & Model")
if st.sidebar.button("Regenerate Data & Retrain Model"):
    generate_dataset(DATA_PATH, rows=360, seed=42)
    model, metrics = train_or_load_model(retrain=True)
    st.sidebar.success("Model retrained on fresh dataset.")

st.title("Robotic Arm Joint Health Monitor")
st.caption("Continuous condition monitoring for industrial manipulator joints (Shoulder, Elbow, Wrist)")

eval_res = evaluate_arm(
    active_scenario,
    model=model,
    maintenance_note=active_scenario.get("maintenance_note", ""),
    previous_readings=active_scenario.get("previous_readings")
)

decision = eval_res["decision"]
if decision == "EMERGENCY_STOP":
    st.error("### 🛑 EMERGENCY STOP ACTIVATED")
    st.write("**Deterministic safety threshold breached. Immediate shutdown initiated independently of ML model.**")
elif decision == "SCHEDULE_MAINTENANCE":
    st.warning("### ⚠️ SCHEDULE MAINTENANCE")
    st.write("**Maintenance inspection required due to high wear risk, warning notes, or sensor faults.**")
elif decision == "MONITOR":
    st.info("### 🔍 STATUS: MONITOR")
    st.write("**Advisory condition: mild wear indicators or caution-level sensor values detected.**")
else:
    st.success("### ✅ STATUS: SAFE")
    st.write("**All joint readings are within normal operating bounds. No safety risks detected.**")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Final Decision", decision)
m2.metric("Predicted Wear Risk", f"{eval_res['wear_probability']:.1%}")
m3.metric("Note Signal", eval_res["note_signal"])
m4.metric("Sensor Integrity", "Suspicious" if eval_res["spoof_suspected"] else "Plausible")

st.markdown("---")
st.subheader("Joint Sensor Readings")

c1, c2, c3 = st.columns(3)

def show_joint(col, name, data, wear_p):
    with col:
        st.markdown(f"#### {name}")
        t = data["torque"]
        temp = data["temperature"]
        vib = data["vibration"]

        t_delta = "CRITICAL" if t >= HARD_LIMITS["torque"] else ("CAUTION" if t >= CAUTION_LIMITS["torque"] else "Normal")
        temp_delta = "CRITICAL" if temp >= HARD_LIMITS["temperature"] else ("CAUTION" if temp >= CAUTION_LIMITS["temperature"] else "Normal")
        vib_delta = "CRITICAL" if vib >= HARD_LIMITS["vibration"] else ("CAUTION" if vib >= CAUTION_LIMITS["vibration"] else "Normal")

        st.metric("Torque", f"{t:.1f} Nm", t_delta, delta_color="inverse" if t_delta != "Normal" else "off")
        st.progress(min(1.0, max(0.0, t / 110.0)))

        st.metric("Temperature", f"{temp:.1f} °C", temp_delta, delta_color="inverse" if temp_delta != "Normal" else "off")
        st.progress(min(1.0, max(0.0, temp / 110.0)))

        st.metric("Vibration", f"{vib:.2f} mm/s", vib_delta, delta_color="inverse" if vib_delta != "Normal" else "off")
        st.progress(min(1.0, max(0.0, vib / 22.0)))

        st.caption(f"Predicted Joint Wear Risk: **{wear_p:.1%}**")

show_joint(c1, "Shoulder (Joint 1)", active_scenario["shoulder"], eval_res["joint_wear_probabilities"].get("shoulder", 0.0))
show_joint(c2, "Elbow (Joint 2)", active_scenario["elbow"], eval_res["joint_wear_probabilities"].get("elbow", 0.0))
show_joint(c3, "Wrist (Joint 3)", active_scenario["wrist"], eval_res["joint_wear_probabilities"].get("wrist", 0.0))

note_text = active_scenario.get("maintenance_note", "")
if note_text:
    st.info(f"**Current Maintenance Note:** \"{note_text}\" (Classification: **{eval_res['note_signal']}**)")

with st.expander("Diagnostic Reason Log", expanded=True):
    for r in eval_res["reasons"]:
        st.write(f"- {r}")

st.markdown("---")

tab_model, tab_notes, tab_spoof, tab_limits = st.tabs([
    "Model Evaluation & Historical Dataset",
    "Maintenance Log Viewer",
    "Spoofed-Sensor Defense",
    "Safety Thresholds & Architecture"
])

with tab_model:
    st.subheader("Random Forest Classifier Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Test Accuracy", f"{metrics['accuracy']:.1%}")
    k2.metric("Precision", f"{metrics['precision']:.3f}")
    k3.metric("Recall", f"{metrics['recall']:.3f}")
    k4.metric("F1 Score", f"{metrics['f1']:.3f}")

    col_cm, col_fi = st.columns(2)
    with col_cm:
        st.write("**Confusion Matrix (Holdout Test Set)**")
        cm_data = pd.DataFrame(
            metrics["confusion_matrix"],
            columns=["Pred: Healthy", "Pred: Wear"],
            index=["Actual: Healthy", "Actual: Wear"]
        )
        st.dataframe(cm_data, width="stretch")

    with col_fi:
        st.write("**Feature Importance**")
        fi_series = pd.Series(metrics["feature_importances"]).sort_values(ascending=False)
        st.bar_chart(fi_series)

    st.write("**Synthetic Dataset Preview (`data/historical_joint_readings.csv`)**")
    if Path(DATA_PATH).exists():
        raw_df = pd.read_csv(DATA_PATH)
        st.dataframe(raw_df.head(20), width="stretch")
        st.caption(f"Showing first 20 of {len(raw_df)} labeled observations (wear_label: 0 = healthy, 1 = wear risk).")

with tab_notes:
    st.subheader("Technician Maintenance Notes (`data/maintenance_log.txt`)")
    logs = load_maintenance_logs()
    if logs:
        st.dataframe(pd.DataFrame(logs)[["date", "arm_id", "note", "signal", "explanation"]], width="stretch")

    st.write("**Test Custom Note**")
    user_test_note = st.text_input("Enter text to test keyword classifier:", "Technician observed excessive vibration and loose mounting bolt.")
    if user_test_note:
        sig, exp = analyze_maintenance_note(user_test_note)
        st.write(f"Result: **{sig}** — {exp}")

with tab_spoof:
    st.subheader("Sensor Defense Heuristics")
    st.write("""
    The defense layer protects against sensor malfunctions and signal spoofing using two deterministic rules:
    1. **Rate-of-Change Limit:** Instantaneous single-sample torque jumps > 35 Nm violate physical motor acceleration constraints.
    2. **Thermodynamic Consistency:** High torque (> 80 Nm) with near-zero vibration (< 0.5 mm/s) and cold temperature (< 35 °C) is physically impossible under load.
    3. **Range Validation:** Negative vibrations or temperatures below -40 °C indicate sensor failure.
    """)
    st.warning("""
    **Defense Limitations:**
    - Slow drift attacks (e.g. raising values by 0.2 Nm per cycle) will not trigger the slew rate check.
    - An attacker who knows the machine model can spoof realistic multi-sensor telemetry.
    - True industrial security requires cryptographic bus authentication (MACsec/TLS) and dual-channel hardware encoders.
    """)

with tab_limits:
    st.subheader("Operating Thresholds")
    threshold_table = pd.DataFrame([
        {"Metric": "Torque (Nm)", "Normal": "< 55.0", "Caution (MONITOR)": "55.0 - 94.9", "Hard Limit (EMERGENCY_STOP)": ">= 95.0"},
        {"Metric": "Temperature (°C)", "Normal": "< 58.0", "Caution (MONITOR)": "58.0 - 89.9", "Hard Limit (EMERGENCY_STOP)": ">= 90.0"},
        {"Metric": "Vibration (mm/s)", "Normal": "< 5.0", "Caution (MONITOR)": "5.0 - 17.9", "Hard Limit (EMERGENCY_STOP)": ">= 18.0"}
    ])
    st.table(threshold_table)
    st.caption("Safety Note: This is an academic simulation for the System Automation course. Real robots require manufacturer limits and ISO 13849 certification.")
