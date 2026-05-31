import pandas as pd
import numpy as np
import joblib
import os


# Load new data
df = pd.read_csv("data/new_data.csv")


# Load preprocessing artifacts
scaler = joblib.load("../artifacts/preprocessing/scaler.pkl")
encoder = joblib.load("../artifacts/preprocessing/encoder.pkl")

num_cols = joblib.load("../artifacts/preprocessing/num_cols.pkl")
cat_cols = joblib.load("../artifacts/preprocessing/cat_cols.pkl")


# Safety check 
missing_num = [col for col in num_cols if col not in df.columns]
missing_cat = [col for col in cat_cols if col not in df.columns]

if missing_num or missing_cat:
    raise ValueError(
        f"Missing columns in new data. "
        f"Numerical missing: {missing_num}, Categorical missing: {missing_cat}"
    )


# Split features
X_num = df[num_cols]
X_cat = df[cat_cols]


# Apply SAME transformations
X_num_scaled = scaler.transform(X_num)
X_cat_encoded = encoder.transform(X_cat)


# Combine features
X_new = np.hstack([X_num_scaled, X_cat_encoded])


# Save processed output
os.makedirs("../artifacts/data", exist_ok=True)

np.save("../artifacts/data/X_new.npy", X_new)

print("Preprocessing complete. Saved X_new.npy")
print("Shape:", X_new.shape)