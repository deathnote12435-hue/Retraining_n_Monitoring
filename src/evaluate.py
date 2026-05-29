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

