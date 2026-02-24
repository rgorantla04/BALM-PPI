#!/usr/bin/env python
"""
Model-1 training script using frozen ESM-2 backbone.

Model-1 is BALM architecture with pre-computed embeddings and frozen backbone.
Supports three split strategies: random, cold_target, and sequence_similarity.

Usage:
    python train_model1.py --config configs/model_1_config.yaml --split cold_target
"""

import os
# Ensure cuBLAS reproducibility for deterministic algorithms (must be set before importing torch)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import argparse
import pickle
import torch
import numpy as np
import pandas as pd
import hashlib
import json
import time
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.reproducibility import setup_reproducibility
from src.utils.config import load_config
from src.utils.visualization import plot_regression
from src.data.loader import (
    load_dataset, get_pkd_bounds, ProteinPairEmbeddingDataset,
    collate_fn_embeddings, generate_and_cache_embeddings
)
from src.data.splits import get_data_splits
from src.data.embeddings import ESM2EmbeddingExtractor
from src.models.architectures import BALMForRegression
from src.models.training import evaluate_model, save_fold_results, save_summary_metrics


def main(args):
    """Main training function for Model-1."""
    
    # Load configuration
    config = load_config(args.config)
    
    # Override split if specified
    if args.split:
        config['data_split']['cv_strategy'] = args.split
    
    # Setup reproducibility
    setup_reproducibility(config['reproducibility']['seed'])
    
    # Setup directories
    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    cache_dir = Path(config['data']['cache_dir'])
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"🧬 {config['description']}")
    print(f"Split Strategy: {config['data_split']['cv_strategy']}")
    print(f"{'='*80}\n")
    
    # Load data
    print("📊 Loading dataset...")
    df = load_dataset(config['data']['dataset_path'])
    
    # Get pKd bounds
    pkd_lower, pkd_upper = get_pkd_bounds(df)
    pkd_bounds = (pkd_lower, pkd_upper)
    print(f"pKd range: [{pkd_lower:.2f}, {pkd_upper:.2f}]")
    
    # Setup device
    device = torch.device(config['device']['type'] if config['device']['type'] != 'auto'
                         else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"🖥️ Using device: {device}")
    # Enforce deterministic behavior for torch where possible
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    
    # Setup embeddings
    print("\n🧮 Setting up embeddings...")
    embedding_extractor = ESM2EmbeddingExtractor(
        model_name=config['model']['model_name'],
        device=str(device)
    )
    
    embedding_cache_path = cache_dir / "model1_embeddings.pkl"
    embedding_dict, embedding_size = generate_and_cache_embeddings(
        df, embedding_extractor, str(embedding_cache_path),
        config['training']['batch_size']
    )
    
    # Get data splits
    split_method = config['data_split']['cv_strategy']
    splits, df_processed = get_data_splits(
        df,
        split_method=split_method,
        n_folds=config['data_split']['n_folds'],
        seed=config['data_split']['seed']
    )

    # --- DIAGNOSTICS: save reproducibility artifacts for debugging ---
    def _file_checksum(path):
        try:
            h = hashlib.md5()
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    diagnostics = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'script': str(Path(__file__).resolve()),
        'config': args.config,
        'seed': config['reproducibility']['seed'],
        'device': str(device),
        'split_method': split_method,
        'n_folds': config['data_split']['n_folds'],
        'dataset_path': config['data']['dataset_path'],
        'dataset_checksum': _file_checksum(config['data']['dataset_path']) if config['data'].get('dataset_path') else None,
        'pkd_bounds': {'lower': float(pkd_lower), 'upper': float(pkd_upper)}
    }

    # embedding cache checksum if exists
    try:
        emb_cache_path = str(embedding_cache_path)
        diagnostics['embedding_cache'] = emb_cache_path
        diagnostics['embedding_cache_checksum'] = _file_checksum(emb_cache_path) if os.path.exists(emb_cache_path) else None
    except Exception:
        diagnostics['embedding_cache'] = None
        diagnostics['embedding_cache_checksum'] = None

    diag_path = results_dir / 'run_diagnostics.json'
    with open(diag_path, 'w') as _f:
        json.dump(diagnostics, _f, indent=2)

    # Save split indices for reproducibility checks
    try:
        splits_save_path = results_dir / 'cv_splits_indices.pkl'
        with open(splits_save_path, 'wb') as _f:
            pickle.dump(splits, _f)
    except Exception as e:
        print(f"Warning: unable to save splits indices: {e}")
    
    print(f"\n🚀 Starting {config['data_split']['n_folds']}-Fold CV with '{split_method}' split")
    
    # Cross-validation loop
    cv_results = []
    all_y_true = []
    all_y_pred = []
    
    for fold, (train_idx, test_idx) in enumerate(splits):
        fold_num = fold + 1
        print(f"\n{'='*60}")
        print(f"FOLD {fold_num}/{config['data_split']['n_folds']}")
        print(f"{'='*60}")
        print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
        
        # Create datasets (use df_processed: seq_sim split resets index)
        train_dataset = ProteinPairEmbeddingDataset(
            df_processed.iloc[train_idx], embedding_dict, pkd_bounds
        )
        test_dataset = ProteinPairEmbeddingDataset(
            df_processed.iloc[test_idx], embedding_dict, pkd_bounds
        )
        
        # Create dataloaders matching notebook exactly: shuffle=True for train, False for test
        # Global torch seed (set by setup_reproducibility) ensures deterministic shuffling
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            collate_fn=collate_fn_embeddings,
            num_workers=config['device']['num_workers']
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size']*2,
            shuffle=False,
            collate_fn=collate_fn_embeddings,
            num_workers=config['device']['num_workers']
        )
        
        # Initialize model
        model = BALMForRegression(
            embedding_size=embedding_size,
            projected_size=config['model']['projected_size'],
            projected_dropout=config['model']['projected_dropout'],
            pkd_bounds=pkd_bounds
        ).to(device)
        
        # Setup optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        print(f"   Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        # Training loop
        best_val_rmse = float('inf')
        patience_counter = 0
        fold_model_path = str(results_dir / f"best_model_fold_{fold_num}.pth")

        for epoch in range(config['training']['epochs']):
            model.train()
            total_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}", leave=False)
            for batch in pbar:
                optimizer.zero_grad()

                # Move tensors to device
                batch_gpu = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                batch_gpu['pdb_groups'] = batch['pdb_groups']
                batch_gpu['subgroups'] = batch['subgroups']
                batch_gpu['source_dataset'] = batch['source_dataset']

                outputs = model(batch_gpu)
                loss = outputs['loss']
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            avg_loss = total_loss / len(train_loader)

            # Evaluate
            val_metrics, _, _, _, _, _ = evaluate_model(model, test_loader, device, pkd_bounds)
            current_val_rmse = val_metrics['rmse']

            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val RMSE: {current_val_rmse:.4f} | "
                  f"Pearson: {val_metrics['pearson']:.4f}")

            # Save best model (matches notebook: torch.save on improvement)
            if current_val_rmse < best_val_rmse:
                best_val_rmse = current_val_rmse
                patience_counter = 0
                torch.save(model.state_dict(), fold_model_path)
                print(f"   -> New best model saved with Val RMSE: {best_val_rmse:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= config['training']['patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Reload best model before final evaluation (matches notebook exactly)
        if os.path.exists(fold_model_path):
            model.load_state_dict(torch.load(fold_model_path, map_location=device))
            print(f"Loaded best model for Fold {fold_num} for final evaluation.")

        # Final evaluation
        fold_metrics, y_true, y_pred, pdb_groups, subgroups, sources = evaluate_model(
            model, test_loader, device, pkd_bounds
        )
        
        cv_results.append(fold_metrics)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        
        # Save fold results
        if config['output']['save_predictions']:
            save_fold_results(fold_num, y_true, y_pred, pdb_groups, subgroups, 
                            sources, str(results_dir))
        
        print(f"Fold {fold_num} Final Metrics:")
        for key, value in fold_metrics.items():
            print(f"  {key.upper()}: {value:.4f}")
    
    # Save summary
    print(f"\n{'='*80}")
    print("📊 Saving results...")
    
    if config['output']['save_predictions']:
        save_summary_metrics(cv_results, str(results_dir))
    
    # Overall metrics
    from src.utils.metrics import calculate_metrics
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    overall_metrics = calculate_metrics(all_y_true, all_y_pred)
    
    print(f"\nOverall Metrics:")
    for key, value in overall_metrics.items():
        print(f"  {key.upper()}: {value:.4f}")
    
    # Plot
    if config['output']['plot_results']:
        plot_regression(
            all_y_true, all_y_pred, overall_metrics,
            title=f"Model-1 ({split_method} split, {config['data_split']['n_folds']}-Fold CV)",
            filename=str(results_dir / "overall_regression.png")
        )
    
    print(f"\n✅ Training complete! Results saved to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Model-1 with frozen ESM-2")
    parser.add_argument("--config", type=str, default="configs/model_1_config.yaml",
                       help="Path to config file")
    parser.add_argument("--split", type=str, 
                       choices=["random", "cold_target", "sequence_similarity"],
                       help="Override CV split strategy")
    
    args = parser.parse_args()
    main(args)
