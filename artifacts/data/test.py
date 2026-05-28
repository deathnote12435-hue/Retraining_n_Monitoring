import numpy as np


X_train_data = np.array([
    [ 0.45, -0.12,  1.21, -0.89],
    [-0.82,  0.94, -0.34,  0.15],
    [ 1.12, -0.56,  0.88,  1.42],
    [-0.11,  0.23, -1.45, -0.67],
    [ 0.67,  1.15,  0.12,  0.34]
])

X_new_data = np.array([
    [ 0.51, -0.08,  1.15,  2.11],
    [-0.75,  0.88, -0.41,  1.95],
    [ 1.05, -0.61,  0.92,  2.54],
    [-0.18,  0.19, -1.38,  1.88],
    [ 0.72,  1.08,  0.05,  2.23]
])

np.save('artifacts/data/X_train.npy', X_train_data)
np.save('artifacts/data/X_new.npy', X_new_data)


print(X_train_data)
print(X_new_data)
