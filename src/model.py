import pandas as pd 
import numpy as np
import json 
import os 

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.pipeline import Pipeline

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# =====================
# CREATE FOLDERS 
# =====================
os.makedirs("artifacts/models", exist_ok=True)
os.makedirs("artifacts/metrics", exist_ok=True)

# =====================
# LOAD DATA 
# =====================
df = pd.read_csv("data/ObesityDataSet.csv")

# ===========================
# SPLIT FEATURES AND TARGET 
# ===========================
X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]

# Encode target labels 
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label mapping 
id = "fix_label_map"
label_map = {
    str(class_name): int(class_id)
    for class_name, class_id in zip(
        label_encoder.classes_, 
        label_encoder.transform(label_encoder.classes_)
    )
}

with open("artifacts/metrics/label_mapping.json", "w") as f:
    json.dump(label_map, f, indent=4)
    
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

# Convert labels to categorical (for neural network)
num_classes = len(np.unique(y_encoded))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# ===============
# BUILD MODEL 
# ===============
model = Sequential()

model.add(Dense(64, activation = 'relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(num_classes, activation = 'softmax'))

# ===============
# COMPILE MODEL 
# ===============
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# ===============
# TRAIN MODEL 
# ===============
history = model.fit(
    X_train, 
    y_train, 
    epochs=25, 
    batch_size=32, 
    validation_split=0.2
)

# ==================
# EVALUATE MODEL 
# ==================
loss, accuracy = model.evaluate(X_test, y_test)

# =============
# SAVE MODEL
# =============
model.save("artifacts/models/model.keras")

# ========================
# SAVE TRAINING HISTORY
# ========================
history_dict = {
    "loss": [float(x) for x in history.history["loss"]], 
    "accuracy": [float(x) for x in history.history["accuracy"]],
    "val_loss": [float(x) for x in history.history["val_loss"]], 
    "val_accuracy": [float(x) for x in history.history["val_accuracy"]]
}

# ========================
# SAVE EVALUATION METRICS
# ========================
evaluation_metrics = {
    "test_loss": float(loss), 
    "test_accuracy": float(accuracy)
}

with open("artifacts/metrics/evaluation_metrics.json", "w") as f:
    json.dump(evaluation_metrics, f, indent=4)
    
print("Training complete")
print("Model saved successfully")




