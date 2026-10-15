import unittest
import numpy as np
from health_monitor import (
    train_or_load_model,
    evaluate_arm,
    make_demo_scenarios,
    generate_dataset,
    detect_sensor_spoof,
    analyze_maintenance_note,
    load_maintenance_logs,
    DATA_PATH
)


class HealthMonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.metrics = train_or_load_model(retrain=True)
        cls.scenarios = make_demo_scenarios()

    def test_01_normal_readings_produce_safe(self):
        scenario = self.scenarios["SAFE"]
        result = evaluate_arm(
            scenario,
            self.model,
            scenario.get("maintenance_note", ""),
            scenario.get("previous_readings")
        )
        self.assertEqual(result["decision"], "SAFE")
        self.assertFalse(result["spoof_suspected"])
        self.assertFalse(result["safety_stop_triggered"])
        self.assertLess(result["wear_probability"], 0.35)

    def test_02_moderate_wear_risk_produces_monitor(self):
        scenario = self.scenarios["MONITOR"]
        result = evaluate_arm(
            scenario,
            self.model,
            scenario.get("maintenance_note", ""),
            scenario.get("previous_readings")
        )
        self.assertEqual(result["decision"], "MONITOR")
        self.assertFalse(result["spoof_suspected"])
        self.assertFalse(result["safety_stop_triggered"])

    def test_03_maintenance_warnings_produce_schedule_maintenance(self):
        scenario = self.scenarios["SCHEDULE_MAINTENANCE"]
        result = evaluate_arm(
            scenario,
            self.model,
            scenario.get("maintenance_note", ""),
            scenario.get("previous_readings")
        )
        self.assertEqual(result["decision"], "SCHEDULE_MAINTENANCE")
        self.assertFalse(result["safety_stop_triggered"])
        self.assertEqual(result["note_signal"], "WARNING")

        urgent_scenario = {
            "shoulder": {"torque": 40.0, "temperature": 45.0, "vibration": 2.5},
            "elbow": {"torque": 42.0, "temperature": 44.0, "vibration": 2.4},
            "wrist": {"torque": 38.0, "temperature": 40.0, "vibration": 2.0},
            "maintenance_note": "Technician observed smoke and burning smell from base."
        }
        urgent_result = evaluate_arm(urgent_scenario, self.model, urgent_scenario["maintenance_note"])
        self.assertEqual(urgent_result["decision"], "SCHEDULE_MAINTENANCE")
        self.assertEqual(urgent_result["note_signal"], "URGENT")

    def test_04_dangerous_sensor_reading_produces_emergency_stop(self):
        scenario = self.scenarios["EMERGENCY_STOP"]
        result = evaluate_arm(
            scenario,
            self.model,
            scenario.get("maintenance_note", ""),
            scenario.get("previous_readings")
        )
        self.assertEqual(result["decision"], "EMERGENCY_STOP")
        self.assertTrue(result["safety_stop_triggered"])
        self.assertEqual(result["wear_probability"], 1.0)
        self.assertTrue(any("exceeded hard limit" in r for r in result["reasons"]))

        high_torque = {
            "shoulder": {"torque": 98.5, "temperature": 40.0, "vibration": 2.0},
            "elbow": {"torque": 40.0, "temperature": 40.0, "vibration": 2.0},
            "wrist": {"torque": 30.0, "temperature": 35.0, "vibration": 1.5}
        }
        res_torque = evaluate_arm(high_torque, self.model)
        self.assertEqual(res_torque["decision"], "EMERGENCY_STOP")

        high_vib = {
            "shoulder": {"torque": 40.0, "temperature": 40.0, "vibration": 2.0},
            "elbow": {"torque": 40.0, "temperature": 40.0, "vibration": 2.0},
            "wrist": {"torque": 30.0, "temperature": 35.0, "vibration": 22.0}
        }
        res_vib = evaluate_arm(high_vib, self.model)
        self.assertEqual(res_vib["decision"], "EMERGENCY_STOP")

    def test_05_emergency_stop_independent_of_ml_model(self):
        scenario = self.scenarios["EMERGENCY_STOP"]
        result = evaluate_arm(scenario, model=None, maintenance_note="Normal note")
        self.assertEqual(result["decision"], "EMERGENCY_STOP")
        self.assertTrue(result["safety_stop_triggered"])
        self.assertIn("independently of machine learning model", result["reasons"][0])

    def test_06_spoofing_scenario_detected(self):
        scenario = self.scenarios["SPOOF"]
        result = evaluate_arm(
            scenario,
            self.model,
            scenario.get("maintenance_note", ""),
            scenario.get("previous_readings")
        )
        self.assertTrue(result["spoof_suspected"])
        self.assertEqual(result["decision"], "SCHEDULE_MAINTENANCE")
        self.assertTrue(any("torque shifted abruptly" in r for r in result["reasons"]))
        self.assertTrue(any("contradictory state" in r for r in result["reasons"]))

        spoofed_arm = {
            "shoulder": {"torque": 85.0, "temperature": 25.0, "vibration": 0.2},
            "elbow": {"torque": 40.0, "temperature": 45.0, "vibration": 2.0},
            "wrist": {"torque": 35.0, "temperature": 40.0, "vibration": 2.0}
        }
        flag, reasons = detect_sensor_spoof(spoofed_arm)
        self.assertTrue(flag)
        self.assertTrue(len(reasons) > 0)

    def test_07_invalid_or_missing_sensor_values_handled_safely(self):
        missing_joint_arm = {
            "shoulder": {"torque": 40.0, "temperature": 45.0, "vibration": 2.0},
            "wrist": {"torque": 35.0, "temperature": 40.0, "vibration": 2.0}
        }
        res_missing = evaluate_arm(missing_joint_arm, self.model)
        self.assertEqual(res_missing["decision"], "SCHEDULE_MAINTENANCE")
        self.assertTrue(res_missing["spoof_suspected"])
        self.assertTrue(any("Missing data block for elbow" in r for r in res_missing["reasons"]))

        nan_val_arm = {
            "shoulder": {"torque": np.nan, "temperature": 45.0, "vibration": 2.0},
            "elbow": {"torque": 40.0, "temperature": 45.0, "vibration": 2.0},
            "wrist": {"torque": 35.0, "temperature": None, "vibration": 2.0}
        }
        res_nan = evaluate_arm(nan_val_arm, self.model)
        self.assertEqual(res_nan["decision"], "SCHEDULE_MAINTENANCE")
        self.assertTrue(res_nan["spoof_suspected"])
        self.assertTrue(any("Invalid or missing numeric reading" in r for r in res_nan["reasons"]))

        res_none = evaluate_arm(None, self.model)
        self.assertEqual(res_none["decision"], "SCHEDULE_MAINTENANCE")
        self.assertTrue(res_none["spoof_suspected"])

    def test_08_dataset_and_model_properties(self):
        df = generate_dataset(rows=350, seed=123)
        self.assertEqual(len(df), 350)
        self.assertIn("wear_label", df.columns)
        self.assertIn("torque_nm", df.columns)
        self.assertGreaterEqual(self.metrics["accuracy"], 0.85)
        self.assertGreaterEqual(self.metrics["rows"], 300)

    def test_09_maintenance_logs_parsing(self):
        logs = load_maintenance_logs()
        self.assertGreaterEqual(len(logs), 8)
        signals = {item["signal"] for item in logs}
        self.assertIn("NORMAL", signals)
        self.assertIn("WARNING", signals)


if __name__ == "__main__":
    unittest.main()
