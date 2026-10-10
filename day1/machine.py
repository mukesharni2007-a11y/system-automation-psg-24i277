import time
import random

LOG_FILE = "day1_log1.txt"


# ---------------------------------------------------------------------------
# STEP 1 & STEP 6 — Generate simulated sensor data
# ---------------------------------------------------------------------------

def generate_sensor_data():
    """Return simulated sensor readings."""
    return {
        "temperature": random.randint(50, 95),
        "pressure": round(random.uniform(1.0, 5.0), 2),
        "vibration": round(random.uniform(0.1, 5.0), 2),
        "current": round(random.uniform(5.0, 15.0), 2)
    }


# ---------------------------------------------------------------------------
# MACHINE LIST
# ---------------------------------------------------------------------------

MACHINES = [
    "CNC Machine",
    "Conveyor Motor",
    "Hydraulic Press",
    "Air Compressor",
    "Industrial Pump"
]


# ---------------------------------------------------------------------------
# STEP 2, STEP 7 & STEP 8 — Automation rules
# ---------------------------------------------------------------------------

RULES = [
    {
        "name": "EMERGENCY_HIGH_TEMP",
        "priority": 1,
        "condition": lambda r: r["temperature"] > 90,
        "action": "EMERGENCY_STOP"
    },

    {
        "name": "HIGH_VIBRATION",
        "priority": 2,
        "condition": lambda r: r["vibration"] > 4.0,
        "action": "STOP_MACHINE"
    },

    {
        "name": "HIGH_CURRENT",
        "priority": 3,
        "condition": lambda r: r["current"] > 13,
        "action": "REDUCE_LOAD"
    },

    {
        "name": "HIGH_TEMPERATURE",
        "priority": 4,
        "condition": lambda r: r["temperature"] > 80,
        "action": "COOL_MACHINE"
    },

    {
        "name": "HIGH_PRESSURE",
        "priority": 5,
        "condition": lambda r: r["pressure"] > 4.0,
        "action": "CHECK_PRESSURE"
    }
]


# ---------------------------------------------------------------------------
# STEP 3 — Detect machine state from the rules
# ---------------------------------------------------------------------------

def evaluate_rules(reading, rules):
    """
    Return (rule_name, action) for the first matching rule.
    Rules are checked in priority order.
    """

    sorted_rules = sorted(rules, key=lambda r: r["priority"])

    for rule in sorted_rules:
        if rule["condition"](reading):
            return rule["name"], rule["action"]

    return "NO_RULE_FIRED", "CONTINUE_NORMAL"


# ---------------------------------------------------------------------------
# STEP 4 — Trigger an action when a rule fires
# ---------------------------------------------------------------------------

def take_action(action):
    """Return a human-readable message for the action."""

    actions = {
        "EMERGENCY_STOP": "Emergency stop activated!",
        "STOP_MACHINE": "Machine stopped due to unsafe condition.",
        "REDUCE_LOAD": "Machine load reduced.",
        "COOL_MACHINE": "Cooling system activated.",
        "CHECK_PRESSURE": "Pressure is high. Inspection required.",
        "CONTINUE_NORMAL": "Machine operating normally."
    }

    return actions.get(action, f"No handler for action: {action}")


# ---------------------------------------------------------------------------
# STEP 5 — Log every decision made
# ---------------------------------------------------------------------------

def log_decision(machine, reading, rule_name, action):
    """Append one line to the log file."""

    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(
            f"Machine: {machine} | "
            f"Reading: {reading} | "
            f"Rule: {rule_name} | "
            f"Action: {action}\n"
        )


# ---------------------------------------------------------------------------
# NOVELTY — Calculate risk score for each machine
# ---------------------------------------------------------------------------

def calculate_risk(rule_name):
    """Assign a risk score based on the detected condition."""

    risk_scores = {
        "EMERGENCY_HIGH_TEMP": 100,
        "HIGH_VIBRATION": 80,
        "HIGH_CURRENT": 60,
        "HIGH_TEMPERATURE": 50,
        "HIGH_PRESSURE": 40,
        "NO_RULE_FIRED": 0
    }

    return risk_scores.get(rule_name, 0)


# ---------------------------------------------------------------------------
# NOVELTY — Find machine requiring immediate attention
# ---------------------------------------------------------------------------

def find_priority_machine(machine_data):
    """Return the machine with the highest risk."""

    return max(machine_data, key=lambda x: x["risk"])


# ---------------------------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------------------------

def main():

    for tick in range(1, 11):

        machine_data = []

        print("\n" + "=" * 75)
        print(f"FACTORY MONITORING - TICK {tick:02d}")
        print("=" * 75)

        # Check all 5 machines
        for machine in MACHINES:

            reading = generate_sensor_data()

            rule_name, action = evaluate_rules(reading, RULES)

            outcome = take_action(action)

            risk = calculate_risk(rule_name)

            machine_data.append({
                "machine": machine,
                "reading": reading,
                "rule": rule_name,
                "action": action,
                "risk": risk
            })

            print(f"\nMachine : {machine}")
            print(f"SENSE   : {reading}")
            print(f"THINK   : {rule_name}")
            print(f"RISK    : {risk}/100")
            print(f"ACT     : {outcome}")

            log_decision(
                machine,
                reading,
                rule_name,
                action
            )

        # ---------------------------------------------------------------
        # Find the machine requiring immediate attention
        # ---------------------------------------------------------------

        priority_machine = find_priority_machine(machine_data)

        print("\n" + "-" * 75)
        print("FACTORY PRIORITY ANALYSIS")
        print("-" * 75)

        if priority_machine["risk"] >= 80:

            print(
                f"!!! IMMEDIATE ACTION REQUIRED !!!"
            )

            print(
                f"Machine : {priority_machine['machine']}"
            )

            print(
                f"Risk    : {priority_machine['risk']}/100"
            )

            print(
                f"Problem : {priority_machine['rule']}"
            )

            print(
                f"Action  : {priority_machine['action']}"
            )

        elif priority_machine["risk"] > 0:

            print(
                f"ATTENTION REQUIRED"
            )

            print(
                f"Machine : {priority_machine['machine']}"
            )

            print(
                f"Risk    : {priority_machine['risk']}/100"
            )

        else:

            print("All machines are operating normally.")

        time.sleep(0.5)


if __name__ == "__main__":
    main()