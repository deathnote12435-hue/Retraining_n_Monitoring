# ===================================
# BASELINE (from production model)
# ===================================
BASELINE_ACCURACY = 0.9456
BASELINE_F1 = 0.9454

# ===================
# ALLOWED DROPS
# ===================
ACCURACY_DROP_THRESHOLD = 0.05 #5%
F1_DROP_THRESHOLD = 0.05

# ===================================
# MINIMUM ACCEPTABLE VALUES
# ===================================
MIN_ACCURACY = 0.90
MIN_F1_SCORE = 0.90

# ===================================
# MONITORING RULES FUNCTION 
# ===================================
def check_model_health(current_metrics):
    """
    Check if model performance is within acceptable limits. 
    """
    
    accuracy = current_metrics.get("accuracy", 0)
    f1 = current_metrics.get("f1_score", 0)
    
    alerts = []
    
    # ---------------------------
    # Rule 1: Minimum threshold
    # ---------------------------
    if accuracy < MIN_ACCURACY: 
        alerts.append("Accuracy below minimum threshold")
    if f1 < MIN_F1_SCORE: 
        alerts.append("F1-score below minimum threshold")
    
    # ---------------------------
    # Rule 2: Performance Drop
    # ---------------------------
    if (BASELINE_ACCURACY - accuracy) > ACCURACY_DROP_THRESHOLD: 
        alerts.append("Significant accuracy drop detected")
    if (BASELINE_F1 - f1) > F1_DROP_THRESHOLD: 
        alerts.append("Significant F1-score drop detected")
    
    # -----------
    # Result
    # -----------
    is_healthy = len(alerts) == 0
    
    return {
        "is_healthy": is_healthy, 
        "alerts": alerts
    }
        
    
    
        
    