import numpy as np
import json
import logging
import subprocess
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import load_model
from datetime import datetime
import os
from monitoring_rules import check_model_health

# Logging setup
logging.basicConfig(
    filename="artifacts/logs/monitoring_model.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Starting MODEL-LEVEL monitoring pipeline...")

# Load artifacts
try:
    X_test = np.load("artifacts/data/X_test.npy")
    y_test = np.load("artifacts/data/y_test.npy")
    X_new = np.load("artifacts/data/X_new.npy")

    model = load_model("artifacts/models/model.keras")

    logging.info("Artifacts loaded successfully.")

except Exception as e:
    logging.error(f"Error loading artifacts: {e}")
    raise e

# Prediction Drift
try:
    y_pred_test = np.argmax(model.predict(X_test), axis=1)
    y_pred_new = np.argmax(model.predict(X_new), axis=1)

    unique_classes = np.unique(np.concatenate([y_pred_test, y_pred_new]))
    pred_drift = {}

    for cls in unique_classes:
        test_freq = np.mean(y_pred_test == cls)
        new_freq = np.mean(y_pred_new == cls)

        pred_drift[int(cls)] = {
            "test_distribution": float(test_freq),
            "new_distribution": float(new_freq),
            "difference": float(abs(test_freq - new_freq))
        }

    logging.info("Prediction drift computed.")

except Exception as e:
    logging.error(f"Prediction drift error: {e}")
    raise e

# Accuracy Drift
try:
    old_accuracy = accuracy_score(y_test, y_pred_test)

    # Compute F1 for health checks
    old_f1 = f1_score(y_test, y_pred_test, average="weighted")

    if os.path.exists("artifacts/data/y_new.npy"):
        y_new = np.load("artifacts/data/y_new.npy")
        new_accuracy = accuracy_score(y_new, y_pred_new)
        new_f1 = f1_score(y_new, y_pred_new, average="weighted")
    else:
        new_accuracy = None
        new_f1 = None

    logging.info("Accuracy drift computed.")

except Exception as e:
    logging.error(f"Accuracy drift error: {e}")
    raise e

# Model Health Check (using monitoring_rules.py)
current_metrics = {
    "accuracy": new_accuracy if new_accuracy is not None else old_accuracy,
    "f1_score": new_f1 if new_f1 is not None else old_f1
}

health = check_model_health(current_metrics)
is_healthy = health["is_healthy"]
alerts = health["alerts"]

logging.info(f"Model health: {is_healthy}, Alerts: {alerts}")

# Determine retrain trigger
retrain_triggered = not is_healthy

logging.info(f"Retrain triggered: {retrain_triggered}")

# Call retrain.py if needed
if retrain_triggered:
    logging.info("Calling retrain.py...")
    try:
        subprocess.run(["python", "retrain.py"], check=True)
        logging.info("Retrain script executed successfully.")
    except Exception as e:
        logging.error(f"Error calling retrain.py: {e}")

# Save monitoring_metrics.json
monitoring_output = {
    "timestamp": datetime.now().isoformat(),
    "old_accuracy": float(old_accuracy),
    "old_f1": float(old_f1),
    "new_accuracy": float(new_accuracy) if new_accuracy is not None else "N/A",
    "new_f1": float(new_f1) if new_f1 is not None else "N/A",
    "prediction_drift": pred_drift,
    "model_health": health,
    "retrain_recommended": retrain_triggered
}

os.makedirs("artifacts/metrics", exist_ok=True)

with open("artifacts/metrics/monitoring_metrics.json", "w") as f:
    json.dump(monitoring_output, f, indent=4)

logging.info("monitoring_metrics.json saved.")

# Save drift_report.json
drift_report = {
    "timestamp": datetime.now().isoformat(),
    "prediction_drift": pred_drift,
    "accuracy_drift": {
        "old_accuracy": float(old_accuracy),
        "new_accuracy": float(new_accuracy) if new_accuracy is not None else "N/A"
    },
    "f1_drift": {
        "old_f1": float(old_f1),
        "new_f1": float(new_f1) if new_f1 is not None else "N/A"
    },
    "model_health": health,
    "retrain_recommended": retrain_triggered
}

with open("artifacts/metrics/drift_report.json", "w") as f:
    json.dump(drift_report, f, indent=4)

logging.info("drift_report.json saved.")

# Print summary
print("\n MODEL-LEVEL MONITORING SUMMARY")
print(f"Old Accuracy: {old_accuracy:.4f}")
print(f"Old F1 Score: {old_f1:.4f}")
print(f"New Accuracy: {new_accuracy if new_accuracy is not None else 'N/A'}")
print(f"New F1 Score: {new_f1 if new_f1 is not None else 'N/A'}")

print("\nPrediction Drift:")
for cls, stats in pred_drift.items():
    print(f"Class {cls}: {stats}")

print("\nModel Health:", health)
print(f"Retrain Recommended: {retrain_triggered}")

print("\nMonitoring complete. Reports saved to artifacts/metrics/")
