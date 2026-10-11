import time
import random
import joblib
import pandas as pd


# ============================================================
# LOAD RANDOM FOREST MODEL
# ============================================================

model = joblib.load("random_forest_model.pkl")

print("Random Forest AI model loaded successfully!")


# ============================================================
# GENERATE SENSOR DATA
# Same ranges used by the ML dataset
# ============================================================

def generate_sensor_data():

    return {
        "temperature": random.randint(50, 95),

        "pressure": round(
            random.uniform(4.5, 10.0), 2
        ),

        "vibration": round(
            random.uniform(0.3, 5.0), 2
        ),

        "rpm": random.randint(2500, 3200),

        "current": round(
            random.uniform(7.0, 16.0), 2
        )
    }


# ============================================================
# RULE-BASED STATE
#
# THESE RULES MUST MATCH train_model.py
# ============================================================

def get_machine_state(reading):

    temperature = reading["temperature"]
    pressure = reading["pressure"]
    vibration = reading["vibration"]
    rpm = reading["rpm"]
    current = reading["current"]


    # -----------------------------
    # FAULT
    # -----------------------------

    if (
        temperature >= 90
        or pressure >= 9.0
        or vibration >= 4.0
        or rpm < 2600
        or rpm > 3150
        or current >= 15.0
    ):

        return "Fault"


    # -----------------------------
    # WARNING
    # -----------------------------

    elif (
        temperature >= 80
        or pressure >= 7.0
        or vibration >= 2.5
        or rpm < 2750
        or rpm > 3050
        or current >= 13.5
    ):

        return "Warning"


    # -----------------------------
    # NORMAL
    # -----------------------------

    else:

        return "Normal"


# ============================================================
# RULE NAME
# ============================================================

def get_rule_name(reading):

    temperature = reading["temperature"]
    pressure = reading["pressure"]
    vibration = reading["vibration"]
    rpm = reading["rpm"]
    current = reading["current"]


    # FAULT

    if temperature >= 90:
        return "EMERGENCY_HIGH_TEMP"

    elif pressure >= 9.0:
        return "CRITICAL_HIGH_PRESSURE"

    elif vibration >= 4.0:
        return "CRITICAL_HIGH_VIBRATION"

    elif rpm < 2600 or rpm > 3150:
        return "CRITICAL_ABNORMAL_RPM"

    elif current >= 15.0:
        return "CRITICAL_HIGH_CURRENT"


    # WARNING

    elif temperature >= 80:
        return "HIGH_TEMPERATURE"

    elif pressure >= 7.0:
        return "HIGH_PRESSURE"

    elif vibration >= 2.5:
        return "HIGH_VIBRATION"

    elif rpm < 2750 or rpm > 3050:
        return "ABNORMAL_RPM"

    elif current >= 13.5:
        return "HIGH_CURRENT"


    return "NO_RULE_FIRED"


# ============================================================
# ACTION
# ============================================================

def take_action(state):

    if state == "Fault":

        return "EMERGENCY_STOP"

    elif state == "Warning":

        return "REDUCE_LOAD"

    else:

        return "CONTINUE_NORMAL"


# ============================================================
# AI PREDICTION
# ============================================================

def ai_prediction(reading):

    input_data = pd.DataFrame({

        "Temperature": [
            reading["temperature"]
        ],

        "Pressure": [
            reading["pressure"]
        ],

        "Vibration": [
            reading["vibration"]
        ],

        "RPM": [
            reading["rpm"]
        ],

        "Current": [
            reading["current"]
        ]
    })


    prediction = model.predict(
        input_data
    )[0]


    probabilities = model.predict_proba(
        input_data
    )[0]


    confidence = max(
        probabilities
    ) * 100


    return prediction, confidence


# ============================================================
# MAIN
# ============================================================

def main():

    total = 10

    matches = 0
    differences = 0


    print("=" * 65)
    print("       SMART FACTORY AI vs RULE-BASED SYSTEM")
    print("=" * 65)


    for tick in range(1, total + 1):


        # -----------------------------------------
        # SAME SENSOR READING
        # -----------------------------------------

        reading = generate_sensor_data()


        # -----------------------------------------
        # RULE-BASED SYSTEM
        # -----------------------------------------

        logic_state = get_machine_state(
            reading
        )

        rule_name = get_rule_name(
            reading
        )

        action = take_action(
            logic_state
        )


        # -----------------------------------------
        # AI SYSTEM
        # -----------------------------------------

        ai_state, confidence = ai_prediction(
            reading
        )


        # -----------------------------------------
        # COMPARE
        # -----------------------------------------

        if logic_state == ai_state:

            comparison = "MATCH"
            matches += 1

        else:

            comparison = "DIFFERENT"
            differences += 1


        # -----------------------------------------
        # DISPLAY
        # -----------------------------------------

        print("\n" + "=" * 65)

        print(f"TICK {tick}")

        print("=" * 65)


        print("\nSENSOR READINGS")

        print(
            f"Temperature : "
            f"{reading['temperature']}"
        )

        print(
            f"Pressure    : "
            f"{reading['pressure']}"
        )

        print(
            f"Vibration   : "
            f"{reading['vibration']}"
        )

        print(
            f"RPM         : "
            f"{reading['rpm']}"
        )

        print(
            f"Current     : "
            f"{reading['current']}"
        )


        # -----------------------------------------
        # LOGIC RESULT
        # -----------------------------------------

        print("\nRULE-BASED SYSTEM")

        print(
            f"Rule        : {rule_name}"
        )

        print(
            f"State       : {logic_state}"
        )

        print(
            f"Action      : {action}"
        )


        # -----------------------------------------
        # AI RESULT
        # -----------------------------------------

        print("\nRANDOM FOREST AI")

        print(
            f"Prediction  : {ai_state}"
        )

        print(
            f"Confidence  : {confidence:.2f}%"
        )


        # -----------------------------------------
        # COMPARISON
        # -----------------------------------------

        print("\nCOMPARISON")

        print(
            f"Logic State : {logic_state}"
        )

        print(
            f"AI State    : {ai_state}"
        )

        print(
            f"Result      : {comparison}"
        )


        time.sleep(0.5)


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n\n")

    print("=" * 65)
    print("                 FINAL COMPARISON")
    print("=" * 65)


    print(
        f"Total Readings       : {total}"
    )

    print(
        f"Matching             : {matches}"
    )

    print(
        f"Different            : {differences}"
    )


    agreement = (
        matches / total
    ) * 100


    print(
        f"AI vs Logic Agreement: "
        f"{agreement:.2f}%"
    )


    print("=" * 65)


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()