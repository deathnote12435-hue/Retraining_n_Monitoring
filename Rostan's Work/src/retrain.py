import pandas as pd 
import numpy as np
import json 
import os 

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.metrics import accuracy_score, f1_score

from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
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

# ===============
# PREPROCESS 
# ===============
categorical_cols = X.select_dtypes(include=["object"]).columns
numeric_cols = X.select_dtypes(exclude=["object"]).columns

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

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# ===========================
# LOAD OLD MODEL (if exists)
# ===========================
old_model_path = "../artifacts/models/model.keras"

old_accuracy = 0

if os.path.exists(old_model_path):
    old_model = load_model(old_model_path)
    old_loss, old_accuracy = old_model.evaluate(X_test, y_test, verbose=0)
    print(f"Old model accuracy: {old_accuracy}")
    
# =====================
# BUILD NEW MODEL 
# =====================
model = Sequential([
    Dense(64, activation = 'relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation = 'relu'),
    Dense(num_classes, activation = 'softmax')
])

model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# ===============
# TRAIN NEW MODEL 
# ===============
history = model.fit(
    X_train, 
    y_train, 
    epochs=20, 
    batch_size=32, 
    validation_split=0.2, 
    verbose=0
)

# ======================
# EVALUATION NEW MODEL 
# ======================
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"New model accuracy: {accuracy}")

# ======================
# CALCULATE F1 SCORES 
# ======================
true_classes = np.argmax(y_test, axis=1)

# ------- OLD MODEL F1 --------
if os.path.exists(old_model_path):
    old_predictions = old_model.predict(X_test, verbose=0)
    old_pred_classes = np.argmax(old_predictions, axis=1)
    
    old_f1 = f1_score(
        true_classes, 
        old_pred_classes, 
        average="weighted"
)

else: 
    old_f1 = 0
    
# -------- NEW MODEL F1 --------
new_predictions = model.predict(X_test, verbose=0)
new_pred_classes = np.argmax(new_predictions, axis=1)

new_f1 = f1_score(
        true_classes, 
        new_pred_classes, 
        average="weighted"
)

# ============================
# RETRAINING DECISION LOGIC 
# ============================
from metrics_comparator import should_replace_model

old_metrics = {
    "accuracy": old_accuracy, 
    "f1_score": old_f1
}

new_metrics = {
    "accuracy": accuracy, 
    "f1_score": new_f1
}

if should_replace_model(old_metrics, new_metrics):
    print("New model is better --> saving model")
    model.save("../artifacts/models/model.keras")

    with open("../artifacts/metrics/retrain_metrics.json", "w") as f:
        json.dump(new_metrics, f, indent=4)

else: 
    print("Old model is better --> keeping existing model")
    

    


