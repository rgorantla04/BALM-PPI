"""
Metrics computation for protein-protein binding affinity prediction.
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Concordance Index (C-index) for regression.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Concordance index value between 0 and 1
    """
    try:
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        n = len(y_true)
        
        if n < 2:
            return 1.0
        
        concordant, discordant, tied_pred = 0, 0, 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if y_true[i] == y_true[j]:
                    continue
                if y_pred[i] == y_pred[j]:
                    tied_pred += 1
                    continue
                if (y_pred[i] > y_pred[j] and y_true[i] > y_true[j]) or \
                   (y_pred[i] < y_pred[j] and y_true[i] < y_true[j]):
                    concordant += 1
                else:
                    discordant += 1
        
        total_pairs = concordant + discordant + tied_pred
        
        if total_pairs == 0:
            return 1.0
        
        return (concordant + 0.5 * tied_pred) / total_pairs
    
    except Exception as e:
        print(f"Error calculating concordance index: {e}")
        return np.nan


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate all evaluation metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary with metrics: rmse, pearson, spearman, ci
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Filter out NaN values
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    
    if len(y_true_masked) < 2:
        return {
            "rmse": np.nan,
            "pearson": np.nan,
            "spearman": np.nan,
            "ci": np.nan
        }
    
    metrics = {
        "rmse": np.sqrt(mean_squared_error(y_true_masked, y_pred_masked)),
        "pearson": pearsonr(y_true_masked, y_pred_masked)[0],
        "spearman": spearmanr(y_true_masked, y_pred_masked)[0],
        "ci": concordance_index(y_true_masked, y_pred_masked)
    }
    
    return metrics
