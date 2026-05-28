from metrics_comparator import should_replace_model

# ============================
# Case 1: NEW MODEL BETTER
# ============================
old_metrics = {
    "accuracy": 0.90, 
    "f1_score": 0.89
}

new_metrics = {
    "accuracy": 0.95, 
    "f1_score": 0.94
}

print("\n--- CASE 1: New model better ---")
print("Should replace:", should_replace_model(old_metrics, new_metrics))

# ============================
# Case 2: OLD MODEL BETTER
# ============================
old_metrics = {
    "accuracy": 0.95, 
    "f1_score": 0.94
}

new_metrics = {
    "accuracy": 0.90, 
    "f1_score": 0.89
}

print("\n--- CASE 2: Old model better ---")
print("Should replace:", should_replace_model(old_metrics, new_metrics))

# ============================
# Case 3: SAME PERFORMANCE
# ============================
old_metrics = {
    "accuracy": 0.92, 
    "f1_score": 0.91
}

new_metrics = {
    "accuracy": 0.92, 
    "f1_score": 0.91
}

print("\n--- CASE 3: Same performance ---")
print("Should replace:", should_replace_model(old_metrics, new_metrics))




