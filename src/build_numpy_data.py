import pandas as pd
import numpy as np
import joblib
import os

# Load datasets
train_df = pd.read_csv("train/train.csv")
test_df = pd.read_csv("test/test.csv")
new_df = pd.read_csv("data/new_data.csv")

target_col = "NObeyesdad"

# Load preprocessing artifacts
scaler = joblib.load("../artifacts/preprocessing/scaler.pkl")
encoder = joblib.load("../artifacts/preprocessing/encoder.pkl")

num_cols = joblib.load("../artifacts/preprocessing/num_cols.pkl")
cat_cols = joblib.load("../artifacts/preprocessing/cat_cols.pkl")

# Feature transformation function
def transform_X(df):
    X_num = scaler.transform(df[num_cols])
    X_cat = encoder.transform(df[cat_cols])
    return np.hstack([X_num, X_cat])

# TRAIN
X_train = transform_X(train_df)
y_train = train_df[target_col].values

# TEST
X_test = transform_X(test_df)
y_test = test_df[target_col].values

# NEW DATA 
X_new = transform_X(new_df)

# Save all numpy arrays
os.makedirs("../artifacts/data", exist_ok=True)

np.save("../artifacts/data/X_train.npy", X_train)
np.save("../artifacts/data/y_train.npy", y_train)

np.save("../artifacts/data/X_test.npy", X_test)
np.save("../artifacts/data/y_test.npy", y_test)

np.save("../artifacts/data/X_new.npy", X_new)

# output
print("Saved all numpy datasets successfully")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("X_new:", X_new.shape)