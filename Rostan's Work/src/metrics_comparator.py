def should_replace_model(old_metrics, new_metrics, threshold=0.0):
    """
    Decide whether new model should replace old model. 
    
    Parameters: 
        old_metrics (dict): metrics from current production model
        new_metrics (dict): metrics from newly trained model 
        threshold (float): minimum improvement required
    
    Returns: 
        bool: True if new model should replace old model
    """
    
    old_acc = old_metrics.get("accuracy", 0)
    new_acc = new_metrics.get("accuracy", 0)
    
    old_f1 = old_metrics.get("f1_score", 0)
    new_f1 = new_metrics.get("f1_score", 0)
    
    # Weighted decision (accuracy + F1 score)
    old_score = (old_acc + old_f1) / 2
    new_score = (new_acc + new_f1) / 2
    
    improvement = new_score - old_score
    
    print(f"Old score: {old_score:.4f}")
    print(f"New score: {new_score:.4f}")
    print(f"Improvement: {improvement:.4f}")
    
    return improvement > threshold