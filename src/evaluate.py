"""
This is an old version of evaluate.py and is no longer in use.
It is being saved as a reference only 

import pandas as pd 
import numpy as np
import json 
import os 

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report, 
    roc_auc_score
)

from tensorflow.keras.models import load_model 
from tensorflow.keras.utils import to_categorical

# =====================
# LOAD DATA 
# =====================
df = pd.read_csv("../data/ObesityDataSet.csv")

X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]

# =====================
# ENCODE TARGET 
# =====================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

num_classes = len(np.unique(y_encoded))

# ===========================
# IDENTIFY COLUMN TYPES  
# ===========================
categorical_cols = X.select_dtypes(include=["object"]).columns
numeric_cols = X.select_dtypes(exclude=["object"]).columns

# =================
# PREPROCESSING 
# =================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols), 
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

X_processed = preprocessor.fit_transform(X)

# ===================
# TRAIN TEST SPLIT 
# ===================
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, 
    y_encoded, 
    test_size=0.2, 
    random_state=42
)

# =============
# LOAD MODEL
# =============
model = load_model("../artifacts/models/model.keras")

# =============
# PREDICTIONS
# =============
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# =============
# METRICS
# =============
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")
conf_matrix = confusion_matrix(y_test, y_pred)

# ROC-AUC (multi-class safe version)
try: 
    roc_auc = roc_auc_score(
        to_categorical(y_test), 
        y_pred_probs, 
        multi_class="ovr"
    )
except: 
    roc_auc = None
    
# ================
# SAVE METRICS
# ================
os.makedirs("../artifacts/metrics", exist_ok=True)

metrics = {
    "accuracy": float(accuracy), 
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc) if roc_auc is not None else None,
    "confusion_matrix": conf_matrix.tolist()
}

with open("../artifacts/metrics/evaluation_full.json", "w") as f:
    json.dump(metrics, f, indent=4)
    
# ================
# PRINT METRICS
# ================
print("\n===== MODEL EVALUATION =====")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)

print("\nSaved to artifacts/metrics/evaluation_full.json")
"""
import numpy as np
import json
import joblib
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from tensorflow.keras.models import load_model
from datetime import datetime
import os

# Logging setup
logging.basicConfig(
    filename="artifacts/logs/evaluation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Starting evaluation pipeline...")

# Load artifacts
try:
    X_test = np.load("artifacts/data/X_test.npy")
    y_test = np.load("artifacts/data/y_test.npy")
    model = load_model("artifacts/models/model.keras")

    with open("artifacts/preprocessing/feature_columns.json", "r") as f:
        feature_info = json.load(f)

    scaler = joblib.load("artifacts/preprocessing/scaler.pkl")
    encoder = joblib.load("artifacts/preprocessing/encoder.pkl")

    logging.info("Artifacts loaded successfully.")

except Exception as e:
    logging.error(f"Error loading artifacts: {e}")
    raise e

# Ensure correct feature order (if needed)
feature_order = feature_info.get("feature_order", [])

if feature_order:
    try:
        # Reorder columns if Tyler provides encoded feature names later
        X_test = X_test[:, :len(feature_order)]
        logging.info("Applied feature ordering from feature_columns.json.")
    except Exception as e:
        logging.warning(f"Feature ordering skipped: {e}")

# Make predictions
try:
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    logging.info("Model predictions generated.")
except Exception as e:
    logging.error(f"Prediction error: {e}")
    raise e

# Compute metrics
try:
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Multi-class ROC-AUC (requires probability matrix)
    roc_auc = roc_auc_score(y_test, y_pred_probs, multi_class="ovr")

    cm = confusion_matrix(y_test, y_pred).tolist()

    logging.info("Evaluation metrics computed.")

except Exception as e:
    logging.error(f"Metric computation error: {e}")
    raise e

# Save evaluation_full.json
evaluation_output = {
    "timestamp": datetime.now().isoformat(),
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "confusion_matrix": cm
}

os.makedirs("artifacts/metrics", exist_ok=True)

with open("artifacts/metrics/evaluation_full.json", "w") as f:
    json.dump(evaluation_output, f, indent=4)

logging.info("evaluation_full.json saved successfully.")

# Print summary to console
print("\n=== MODEL EVALUATION RESULTS ===")
print(f"Accuracy:       {accuracy:.4f}")
print(f"Precision:      {precision:.4f}")
print(f"Recall:         {recall:.4f}")
print(f"F1 Score:       {f1:.4f}")
print(f"ROC-AUC:        {roc_auc:.4f}")
print("\nConfusion Matrix:")
for row in cm:
    print(row)

print("\nEvaluation complete. Results saved to artifacts/metrics/evaluation_full.json.")
