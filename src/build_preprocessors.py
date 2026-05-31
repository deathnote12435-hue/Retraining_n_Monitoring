import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Load training data
df = pd.read_csv("../train/train.csv")

target_col = "NObeyesdad"  

X = df.drop(columns=[target_col])
y = df[target_col]


# Split column types
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns

print("Numerical columns:", list(num_cols))
print("Categorical columns:", list(cat_cols))

# Create preprocessing objects
scaler = StandardScaler()

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False  
)


# Fit on training data ONLY
X_num_scaled = scaler.fit_transform(X[num_cols])

X_cat_encoded = encoder.fit_transform(X[cat_cols])

# Save preprocessing artifacts
os.makedirs("../artifacts/preprocessing", exist_ok=True)

joblib.dump(scaler, "../artifacts/preprocessing/scaler.pkl")
joblib.dump(encoder, "../artifacts/preprocessing/encoder.pkl")

# Save column metadata
joblib.dump(num_cols.tolist(), "../artifacts/preprocessing/num_cols.pkl")
joblib.dump(cat_cols.tolist(), "../artifacts/preprocessing/cat_cols.pkl")

print("Saved scaler, encoder, and column metadata successfully")