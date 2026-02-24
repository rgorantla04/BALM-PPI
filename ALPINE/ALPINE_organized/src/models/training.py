"""
Training and evaluation utilities for ALPINE experiments.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Tuple, List, Any
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW


def train_epoch(model: torch.nn.Module, dataloader: DataLoader, optimizer: AdamW, 
                device: torch.device) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to use
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Move tensors to device
        batch_gpu = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        # Keep metadata lists on CPU
        for key in ['pdb_groups', 'subgroups', 'source_dataset']:
            if key in batch:
                batch_gpu[key] = batch[key]
        
        outputs = model(batch_gpu)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, device: torch.device, 
                  pkd_bounds: Tuple[float, float]) -> Tuple[Dict, np.ndarray, np.ndarray, 
                                                            np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: Evaluation dataloader
        device: Device to use
        pkd_bounds: Tuple of (min_pkd, max_pkd) for scaling
        
    Returns:
        Tuple of (metrics_dict, y_true, y_pred, pdb_groups, subgroups, sources)
    """
    from src.utils.metrics import calculate_metrics
    
    model.eval()
    
    all_labels = []
    all_preds_pkd = []
    all_pdb_groups = []
    all_subgroups = []
    all_sources = []
    
    pkd_lower, pkd_range = pkd_bounds[0], pkd_bounds[1] - pkd_bounds[0]
    
    with torch.no_grad():
        for batch in dataloader:
            batch_gpu = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Add metadata
            batch_gpu['pdb_groups'] = batch['pdb_groups']
            batch_gpu['subgroups'] = batch['subgroups']
            batch_gpu['source_dataset'] = batch['source_dataset']
            
            outputs = model(batch_gpu)
            cosine_sim = outputs['cosine_similarity'].cpu()
            preds_pkd = ((cosine_sim + 1) / 2) * pkd_range + pkd_lower
            
            all_labels.extend(batch['original_pkds'].numpy())
            all_preds_pkd.extend(preds_pkd.numpy())
            all_pdb_groups.extend(batch['pdb_groups'])
            all_subgroups.extend(batch['subgroups'])
            all_sources.extend(batch['source_dataset'])
    
    labels = np.array(all_labels)
    preds = np.array(all_preds_pkd)
    pdb_groups_arr = np.array(all_pdb_groups)
    subgroups_arr = np.array(all_subgroups)
    sources_arr = np.array(all_sources)
    
    metrics = calculate_metrics(labels, preds)
    
    return metrics, labels, preds, pdb_groups_arr, subgroups_arr, sources_arr


def json_converter(obj: Any) -> Any:
    """
    Helper function to convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        Converted object or raises TypeError
    """
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def save_fold_results(fold_num: int, y_true: np.ndarray, y_pred: np.ndarray, 
                     pdb_groups: np.ndarray, subgroups: np.ndarray, sources: np.ndarray,
                     output_dir: str) -> None:
    """
    Save fold prediction results to CSV.
    
    Args:
        fold_num: Fold number
        y_true: Ground truth values
        y_pred: Predictions
        pdb_groups: PDB identifiers
        subgroups: Subgroup labels
        sources: Source dataset labels
        output_dir: Output directory
    """
    fold_output_df = pd.DataFrame({
        'label': y_true,
        'prediction': y_pred,
        'residual': y_true - y_pred,
        'abs_residual': np.abs(y_true - y_pred),
        'PDB': pdb_groups,
        'Subgroup': subgroups,
        'Source Data Set': sources
    })
    
    output_path = os.path.join(output_dir, f"fold_{fold_num}_predictions.csv")
    fold_output_df.to_csv(output_path, index=False)
    print(f"✅ Predictions for Fold {fold_num} saved to {output_path}")


def save_summary_metrics(fold_results: List[Dict], output_dir: str) -> None:
    """
    Save cross-validation summary metrics to CSV.
    
    Args:
        fold_results: List of metrics dictionaries per fold
        output_dir: Output directory
    """
    results_df = pd.DataFrame(fold_results)
    summary_df = pd.DataFrame({
        'Mean': results_df.mean(),
        'Std Dev': results_df.std()
    })
    
    summary_path = os.path.join(output_dir, "cv_summary_metrics.csv")
    summary_df.to_csv(summary_path)
    print(f"✅ CV summary metrics saved to {summary_path}")
    print(summary_df)
