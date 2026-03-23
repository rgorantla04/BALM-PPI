#!/usr/bin/env python
"""
BALM-PPI training script with LoRA fine-tuning.

BALM-PPI uses BALM architecture with LoRA fine-tuning on ESM-2 for efficient adaptation.
Processes sequences directly (not pre-computed embeddings).

Usage:
    python train_balm_ppi.py --config configs/balm_ppi_config.yaml --split cold_target
"""

import os
import sys
import gc
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from typing import List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.reproducibility import setup_reproducibility
from src.utils.config import load_config
from src.utils.metrics import calculate_metrics
from src.utils.visualization import plot_regression
from src.data.loader import load_dataset, get_pkd_bounds
from src.data.splits import get_data_splits
from src.models.architectures import ProteinEmbeddingExtractor, BALMForLoRAFinetuning
from src.models.training import save_fold_results, save_summary_metrics


class ProteinSequenceDataset(Dataset):
    """Dataset for protein sequences with LoRA fine-tuning."""
    
    def __init__(self, dataframe: pd.DataFrame, pkd_bounds: tuple):
        self.data = dataframe.reset_index(drop=True)
        self.pkd_lower, self.pkd_upper = pkd_bounds
        self.pkd_range = self.pkd_upper - self.pkd_lower
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        original_pkd = float(row['Y'])
        scaled_label = ((original_pkd - self.pkd_lower) / self.pkd_range) * 2 - 1
        
        return {
            "protein_sequence": row["Target"],
            "proteina_sequence": row["proteina"],
            "labels": torch.tensor(scaled_label, dtype=torch.float32),
            "original_pkds": torch.tensor(original_pkd, dtype=torch.float32),
            "pdb_groups": row["PDB"],
            "subgroups": row["Subgroup"],
            "source_dataset": row["SourceDataSet"]
        }


def collate_fn_sequences(batch: List[Dict]) -> Dict:
    """Collate function for sequence datasets."""
    return {
        "protein_sequence": [item['protein_sequence'] for item in batch],
        "proteina_sequence": [item['proteina_sequence'] for item in batch],
        "labels": torch.stack([item['labels'] for item in batch]),
        "original_pkds": torch.stack([item['original_pkds'] for item in batch]),
        "pdb_groups": [item['pdb_groups'] for item in batch],
        "subgroups": [item['subgroups'] for item in batch],
        "source_dataset": [item['source_dataset'] for item in batch]
    }


def evaluate_model_lora(model: torch.nn.Module, dataloader: DataLoader, 
                        device: torch.device, pkd_bounds: tuple) -> tuple:
    """Evaluate LoRA model."""
    model.eval()
    
    all_labels = []
    all_preds_pkd = []
    all_pdb_groups = []
    all_subgroups = []
    all_sources = []
    
    pkd_lower, pkd_range = pkd_bounds[0], pkd_bounds[1] - pkd_bounds[0]
    
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(batch)
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


def main(args):
    """Main training function for BALM-PPI."""
    
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
    if config['model']['use_lora']:
        lora_config = config['model'].get('lora', {})
        print(f"LoRA Config: rank={lora_config.get('rank', 8)}, "
              f"alpha={lora_config.get('alpha', 16)}")
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
    
    # Get data splits
    split_method = config['data_split']['cv_strategy']
    splits, df_processed = get_data_splits(
        df,
        split_method=split_method,
        n_folds=config['data_split']['n_folds'],
        seed=config['data_split']['seed']
    )

    print(f"\n🚀 Starting {config['data_split']['n_folds']}-Fold CV with '{split_method}' split")

    lora_cfg = config['model'].get('lora', {})

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
        train_dataset = ProteinSequenceDataset(df_processed.iloc[train_idx], pkd_bounds)
        test_dataset = ProteinSequenceDataset(df_processed.iloc[test_idx], pkd_bounds)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=config['training']['batch_size'],
            shuffle=True, collate_fn=collate_fn_sequences,
            num_workers=config['device']['num_workers']
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config['training']['batch_size']*2,
            shuffle=False, collate_fn=collate_fn_sequences,
            num_workers=config['device']['num_workers']
        )

        # CRITICAL: Load a fresh ESM-2 + LoRA model for each fold (matches notebook exactly)
        print(f"🏗️ Loading fresh ESM-2 and LoRA adapters for Fold {fold_num}...")
        extractor = ProteinEmbeddingExtractor(
            model_name=config['model']['model_name'],
            device=str(device),
            lora_rank=lora_cfg.get('rank', 8),
            lora_alpha=lora_cfg.get('alpha', 16),
            lora_dropout=lora_cfg.get('dropout', 0.1),
            use_lora=config['model']['use_lora']
        )
        esm_model, tokenizer = extractor.get_model_and_tokenizer()

        # Initialize model
        model = BALMForLoRAFinetuning(
            esm_model=esm_model,
            esm_tokenizer=tokenizer,
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

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Trainable params: {trainable_params}")

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

                # Move labels to device (matches notebook)
                batch['labels'] = batch['labels'].to(device)
                batch['original_pkds'] = batch['original_pkds'].to(device)

                outputs = model(batch)
                loss = outputs['loss']
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            avg_loss = total_loss / len(train_loader)

            # Evaluate
            val_metrics, _, _, _, _, _ = evaluate_model_lora(
                model, test_loader, device, pkd_bounds
            )
            current_val_rmse = val_metrics['rmse']

            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val RMSE: {current_val_rmse:.4f} | "
                  f"Pearson: {val_metrics['pearson']:.4f}")

            # Save best model (matches notebook)
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
            print(f"Loading best model for Fold {fold_num} for final evaluation...")
            model.load_state_dict(torch.load(fold_model_path, map_location=device))

        # Final evaluation
        fold_metrics, y_true, y_pred, pdb_groups, subgroups, sources = evaluate_model_lora(
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

        # Clean up fold model to free memory (matches notebook)
        del model, esm_model, extractor
        gc.collect()
        torch.cuda.empty_cache()
    
    # Save summary
    print(f"\n{'='*80}")
    print("📊 Saving results...")
    
    if config['output']['save_predictions']:
        save_summary_metrics(cv_results, str(results_dir))
    
    # Overall metrics
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
            title=f"BALM-PPI ({split_method} split, {config['data_split']['n_folds']}-Fold CV)",
            filename=str(results_dir / "overall_regression.png")
        )
    
    print(f"\n✅ Training complete! Results saved to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BALM-PPI with LoRA fine-tuning")
    parser.add_argument("--config", type=str, default="configs/balm_ppi_config.yaml",
                       help="Path to config file")
    parser.add_argument("--split", type=str,
                       choices=["random", "cold_target", "sequence_similarity"],
                       help="Override CV split strategy")
    
    args = parser.parse_args()
    main(args)
