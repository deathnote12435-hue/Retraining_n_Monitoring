import os
import json
import logging
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from monitoring_rules import check_model_health

# Logging Setup
logging.basicConfig(
    filename='monitoring/logs/model_monitoring.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("Initiating MODEL-LEVEL monitoring pipeline.")

# Load Artifacts
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

# Accuracy and F1 Drift
try:
    old_accuracy = accuracy_score(y_test, y_pred_test)
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
# Model Health Check
current_metrics = {
    "accuracy": new_accuracy if new_accuracy is not None else old_accuracy,
    "f1_score": new_f1 if new_f1 is not None else old_f1
}

health = check_model_health(current_metrics)
is_healthy = health["is_healthy"]
alerts = health["alerts"]

logging.info(f"Model health: {is_healthy}, Alerts: {alerts}")

retrain_triggered = not is_healthy
logging.info(f"Retrain triggered: {retrain_triggered}")

# Retraining
def build_model(input_dim, num_classes):
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

if retrain_triggered:
    logging.info("Starting internal retraining...")

    # Load full dataset
    df = np.load("artifacts/data/X_train.npy")
    y_train = np.load("artifacts/data/y_train.npy")

    input_dim = df.shape[1]
    num_classes = len(np.unique(y_train))

    new_model = build_model(input_dim, num_classes)

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    new_model.fit(
        df, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    # Evaluate new model
    new_test_pred = np.argmax(new_model.predict(X_test), axis=1)
    new_test_acc = accuracy_score(y_test, new_test_pred)
    new_test_f1 = f1_score(y_test, new_test_pred, average="weighted")

    # Compare
    if new_test_acc > old_accuracy:
        new_model.save("artifacts/models/model.keras")
        logging.info("New model outperformed old model. Model replaced.")
    else:
        logging.info("New model did NOT outperform old model. Keeping existing model.")

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
    "summary": {
        "system_status": "CRITICAL" if retrain_triggered else "HEALTHY",
        "retrain_recommended": retrain_triggered
    },
    "prediction_drift": pred_drift,
    "accuracy_drift": {
        "old_accuracy": float(old_accuracy),
        "new_accuracy": float(new_accuracy) if new_accuracy is not None else "N/A"
    },
    "f1_drift": {
        "old_f1": float(old_f1),
        "new_f1": float(new_f1) if new_f1 is not None else "N/A"
    }
}

with open("monitoring/reports/drift_report.json", "w") as f:
    json.dump(drift_report, f, indent=4)

logging.info("drift_report.json saved.")

# GitHub Actions Output Flag 
if "GITHUB_OUTPUT" in os.environ:
    with open(os.environ["GITHUB_OUTPUT"], "a") as go:
        status_value = "true" if retrain_triggered else "false"
        go.write(f"retrain_status={status_value}\n")

# Console Summary
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



