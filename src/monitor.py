import os
import json
import logging
import numpy as np
from scipy.stats import ks_2samp

# Configure system monitoring logging
logging.basicConfig(
    filename='monitoring/logs/drift_detection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_data_drift(X_train, X_new, alpha_threshold=0.05):

    # Applies the Kolmogorov-Smirnov test across continuous feature distributions. Flags drift if the p-value slips below our alpha limit.
   
    drift_detected = False
    feature_report = {}
    num_features = X_train.shape[1]
    
    for i in range(num_features):
        stat, p_value = ks_2samp(X_train[:, i], X_new[:, i])
        is_drifted = bool(p_value < alpha_threshold)
        
        feature_report[f"feature_{i}"] = {
            "ks_statistic": float(stat),
            "p_value": float(p_value),
            "drift_detected": is_drifted
        }
        
        if is_drifted:
            logging.warning(f"🚨 Statistical shift found in Feature_{i}. P-Value: {p_value:.5f}")
            drift_detected = True
            
    return drift_detected, feature_report

def main():
    logging.info("Initiating automated pipeline data drift inspection.")
    
    # Target files generated upstream by Tyler's preprocessing step
    X_train = np.load('artifacts/data/X_train.npy')
    X_new = np.load('artifacts/data/X_new.npy')
    
    drift_triggered, feature_analysis = check_data_drift(X_train, X_new)
    
    # Structural JSON formatting for live dashboard delivery
    drift_report = {
        "summary": {
            "system_status": "CRITICAL" if drift_triggered else "HEALTHY",
            "retrain_recommended": drift_triggered
        },
        "feature_level_metrics": feature_analysis
    }
    
    os.makedirs('monitoring/reports', exist_ok=True)
    with open('monitoring/reports/drift_report.json', 'w') as f:
        json.dump(drift_report, f, indent=4)
        
    # Standardised tracking metrics file for DVC tracking
    monitoring_metrics = {
        "global_drift_status": int(drift_triggered),
        "total_features_checked": len(feature_analysis)
    }
    os.makedirs('artifacts/metrics', exist_ok=True)
    with open('artifacts/metrics/monitoring_metrics.json', 'w') as f:
        json.dump(monitoring_metrics, f, indent=4)
        
    # Communication interface flag passed directly to the GitHub Runner
    if 'GITHUB_OUTPUT' in os.environ:
        with open(os.environ['GITHUB_OUTPUT'], 'a') as go:
            status_value = "true" if drift_triggered else "false"
            go.write(f"retrain_status={status_value}\n")

if __name__ == "__main__":
    main()