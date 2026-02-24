#!/usr/bin/env python
"""
PLMs ablation study training script.

Runs the Model-1 architecture (BALMForRegression / frozen backbone + projection head)
with different protein language models and projection sizes.

Each PLM is tested with projection sizes [256, 512, 1024] using the cold-target
(GroupKFold) split, matching the PLMs ablation notebooks exactly.

Usage:
    # Run all PLMs with all projection sizes:
    python train_plms.py --config configs/plms_config.yaml --plm esm2

    # Run a specific PLM with a specific projection size:
    python train_plms.py --config configs/plms_config.yaml --plm ablang2 --projected_size 512

    # Available PLM keys: esm2, ablang2, esm_c, progen2_small, progen2_medium
"""

import os
import sys
import argparse
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.reproducibility import setup_reproducibility
from src.utils.config import load_config
from src.utils.metrics import calculate_metrics
from src.utils.visualization import plot_regression
from src.data.loader import (
    load_dataset, get_pkd_bounds,
    ProteinPairEmbeddingDataset, collate_fn_embeddings,
    generate_and_cache_embeddings,
)
from src.data.splits import get_data_splits
from src.data.embeddings import get_embedding_extractor
from src.models.architectures import BALMForRegression
from src.models.training import evaluate_model, save_fold_results, save_summary_metrics


def run_single_experiment(config, plm_cfg, projected_size, df, pkd_bounds,
                           embedding_dict, embedding_size, splits, df_processed,
                           device, results_dir):
    """
    Run one CV experiment for a given PLM and projection size.
    Matches the notebook training loop exactly (save/reload best model per fold).
    """
    split_method = config['data_split']['cv_strategy']
    n_folds = config['data_split']['n_folds']
    plm_name = plm_cfg['name']

    exp_name = f"{plm_name}_proj{projected_size}"
    exp_dir = results_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Experiment: {exp_name}")
    print(f"PLM: {plm_name}  |  Projection size: {projected_size}  |  Split: {split_method}")
    print(f"{'='*70}")

    cv_results = []
    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        fold_num = fold + 1
        print(f"\n--- Fold {fold_num}/{n_folds} | Train: {len(train_idx)}, Test: {len(test_idx)} ---")

        train_dataset = ProteinPairEmbeddingDataset(
            df_processed.iloc[train_idx], embedding_dict, pkd_bounds, embedding_size
        )
        test_dataset = ProteinPairEmbeddingDataset(
            df_processed.iloc[test_idx], embedding_dict, pkd_bounds, embedding_size
        )

        batch_size = plm_cfg.get('batch_size', 32)
        num_workers = config['device'].get('num_workers', 0)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, collate_fn=collate_fn_embeddings,
                                  num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size * 2,
                                 shuffle=False, collate_fn=collate_fn_embeddings,
                                 num_workers=num_workers)

        model = BALMForRegression(
            embedding_size=embedding_size,
            projected_size=projected_size,
            projected_dropout=0.1,
            pkd_bounds=pkd_bounds
        ).to(device)

        optimizer = AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        print(f"   Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        best_val_rmse = float('inf')
        patience_counter = 0
        fold_model_path = str(exp_dir / f"best_model_fold_{fold_num}.pth")

        for epoch in range(config['training']['epochs']):
            model.train()
            total_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}",
                        leave=False)
            for batch in pbar:
                optimizer.zero_grad()

                batch_gpu = {k: v.to(device) for k, v in batch.items()
                             if isinstance(v, torch.Tensor)}
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
            val_metrics, _, _, _, _, _ = evaluate_model(model, test_loader, device, pkd_bounds)
            current_val_rmse = val_metrics['rmse']

            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | "
                  f"Val RMSE: {current_val_rmse:.4f} | Pearson: {val_metrics['pearson']:.4f}")

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

        # Reload best model for final evaluation (matches notebook)
        if os.path.exists(fold_model_path):
            model.load_state_dict(torch.load(fold_model_path, map_location=device))
            print(f"Loaded best model for Fold {fold_num}.")

        fold_metrics, y_true, y_pred, pdb_groups, subgroups, sources = evaluate_model(
            model, test_loader, device, pkd_bounds
        )

        print(f"Fold {fold_num} Final Metrics: "
              + ", ".join(f"{k.upper()}={v:.4f}" for k, v in fold_metrics.items()))

        cv_results.append(fold_metrics)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        if config['output']['save_predictions']:
            save_fold_results(fold_num, y_true, y_pred, pdb_groups, subgroups,
                              sources, str(exp_dir))

    # Summary
    save_summary_metrics(cv_results, str(exp_dir))

    all_y_true_arr = np.array(all_y_true)
    all_y_pred_arr = np.array(all_y_pred)
    overall_metrics = calculate_metrics(all_y_true_arr, all_y_pred_arr)

    print(f"\n{exp_name} Overall Metrics:")
    for key, value in overall_metrics.items():
        print(f"  {key.upper()}: {value:.4f}")

    if config['output']['plot_results']:
        plot_regression(
            all_y_true_arr, all_y_pred_arr, overall_metrics,
            title=f"{exp_name} ({split_method}, {n_folds}-Fold CV)",
            filename=str(exp_dir / "overall_regression.png")
        )

    return overall_metrics


def main(args):
    """Main entry point for PLMs ablation study."""

    config = load_config(args.config)
    setup_reproducibility(config['reproducibility']['seed'])

    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(config['data']['cache_dir'])
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"🧬 {config['description']}")
    print(f"{'='*80}\n")

    # Load dataset
    df = load_dataset(config['data']['dataset_path'])
    pkd_lower, pkd_upper = get_pkd_bounds(df)
    pkd_bounds = (pkd_lower, pkd_upper)
    print(f"pKd range: [{pkd_lower:.2f}, {pkd_upper:.2f}]")

    device = torch.device(config['device']['type'] if config['device']['type'] != 'auto'
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"🖥️  Using device: {device}")

    # Get data splits (cold target, same for all PLMs)
    split_method = config['data_split']['cv_strategy']
    splits, df_processed = get_data_splits(
        df,
        split_method=split_method,
        n_folds=config['data_split']['n_folds'],
        seed=config['data_split']['seed']
    )

    # Determine which PLMs to run
    plms_cfg = config['plms']
    plm_keys_to_run = [args.plm] if args.plm else list(plms_cfg.keys())

    # Determine projection sizes
    all_results = {}

    for plm_key in plm_keys_to_run:
        if plm_key not in plms_cfg:
            print(f"⚠️  PLM key '{plm_key}' not found in config. Skipping.")
            continue

        plm_cfg = plms_cfg[plm_key]
        projected_sizes = [args.projected_size] if args.projected_size \
            else plm_cfg['projected_sizes']

        # Generate / load embeddings for this PLM (shared across projection sizes)
        cache_path = cache_dir / f"embeddings_{plm_key}.pkl"

        print(f"\n{'='*80}")
        print(f"PLM: {plm_cfg['name']} ({plm_cfg['model_name']})")
        print(f"{'='*80}")

        try:
            extractor = get_embedding_extractor(
                plm_key=plm_cfg['plm_key'],
                model_name=plm_cfg['model_name'],
                batch_size=plm_cfg.get('batch_size', 32),
                device=str(device)
            )
        except ImportError as e:
            print(f"⚠️  Failed to load {plm_cfg['name']}: {e}")
            print(f"⏭️  Skipping {plm_cfg['name']} and continuing with next PLM.")
            continue

        embedding_dict, embedding_size = generate_and_cache_embeddings(
            df, extractor, str(cache_path), plm_cfg.get('batch_size', 32)
        )

        # Free the extractor (PLM model) from memory before training loop
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Run one CV experiment per projection size
        for proj_size in projected_sizes:
            metrics = run_single_experiment(
                config=config,
                plm_cfg=plm_cfg,
                projected_size=proj_size,
                df=df,
                pkd_bounds=pkd_bounds,
                embedding_dict=embedding_dict,
                embedding_size=embedding_size,
                splits=splits,
                df_processed=df_processed,
                device=device,
                results_dir=results_dir
            )
            all_results[f"{plm_key}_proj{proj_size}"] = metrics

    # Print final summary table
    print(f"\n{'='*80}")
    print("📊 ABLATION SUMMARY")
    print(f"{'='*80}")
    summary_rows = []
    for exp_name, metrics in all_results.items():
        row = {'experiment': exp_name}
        row.update({k: round(v, 4) for k, v in metrics.items()})
        summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).set_index('experiment')
        print(summary_df.to_string())
        summary_df.to_csv(results_dir / "ablation_summary.csv")
        print(f"\nSummary saved to {results_dir / 'ablation_summary.csv'}")

    print(f"\n✅ PLMs ablation study complete! Results in {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PLMs ablation study")
    parser.add_argument("--config", type=str, default="configs/plms_config.yaml",
                        help="Path to PLMs config file")
    parser.add_argument("--plm", type=str,
                        choices=["esm2", "ablang2", "esm_c", "progen2_small", "progen2_medium"],
                        help="Run only this PLM (default: all PLMs in config)")
    parser.add_argument("--projected_size", type=int,
                        choices=[256, 512, 1024],
                        help="Run only this projection size (default: all sizes in config)")

    args = parser.parse_args()
    main(args)
