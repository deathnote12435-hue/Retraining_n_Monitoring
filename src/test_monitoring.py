from monitoring_rules import check_model_health

test_metrics = {
    "accuracy": 0.88, 
    "f1-score": 0.87
}

result = check_model_health(test_metrics)

print("\n--- MONITORING TEST ---")
print(result)