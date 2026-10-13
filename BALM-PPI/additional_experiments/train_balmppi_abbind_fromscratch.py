"""
Train BALM-PPI (ESM-2 + fresh LoRA, NO PPB-Affinity pretraining) completely
from scratch on the exact same 81-complex AB-Bind training budget MVSF-AB
was retrained on (mvsf_train_30pct_seed{42,123,999}.csv), to answer: does
BALM-PPI's architecture (cosine metric-learning + LoRA) have an inherent
data-efficiency advantage independent of its large-scale PPB-Affinity
pretraining, when both models see only these same 81 complexes from a cold
start?

Uses build_model/BALMForLoRAFinetuning from train_multiseed_cv.py unchanged
(build_model loads a fresh, randomly-initialized-LoRA ESM-2 from HuggingFace
each call -- no checkpoint is loaded, so this is a genuine from-scratch run).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent / "BALM-PPI" / "scripts"))
from train_multiseed_cv import (  # noqa: E402
    ProteinPairSequenceDataset,
    collate_fn,
    build_model,
    evaluate_model,
    setup_reproducibility,
)

EPOCHS = 30
PATIENCE = 15
BATCH_SIZE = 1
LR = 1e-4
SEEDS = [42, 123, 999]

results = []
for seed in SEEDS:
    print(f"\n===== Seed {seed}: BALM-PPI from scratch on 81 AB-Bind complexes =====")
    setup_reproducibility(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(f"abbind_fromscratch_train_seed{seed}.csv")
    test_df = pd.read_csv(f"abbind_fromscratch_test_seed{seed}.csv")
    all_y = pd.concat([train_df["Y"], test_df["Y"]])
    pkd_bounds = (all_y.min(), all_y.max())
    print(f"  train={len(train_df)}  test={len(test_df)}  pkd_bounds={pkd_bounds}")

    # Carve a genuine validation split from the 81 training complexes (80/20),
    # same convention as BALM-PPI's own real few-shot protocol -- early
    # stopping never touches the test set.
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(train_df))
    n_val = max(1, int(0.2 * len(train_df)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    sub_train_df = train_df.iloc[tr_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)
    print(f"  sub_train={len(sub_train_df)}  val={len(val_df)}")

    train_loader = DataLoader(
        ProteinPairSequenceDataset(sub_train_df, pkd_bounds),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        ProteinPairSequenceDataset(val_df, pkd_bounds),
        batch_size=4, shuffle=False, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        ProteinPairSequenceDataset(test_df, pkd_bounds),
        batch_size=4, shuffle=False, collate_fn=collate_fn,
    )

    model = build_model(device, pkd_bounds)  # fresh ESM-2 + fresh LoRA, no checkpoint
    optimizer = AdamW(model.parameters(), lr=LR)

    best_val_rmse = float("inf")
    best_state = None
    patience_counter = 0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_metrics, *_ = evaluate_model(model, val_loader, device, pkd_bounds)
        print(f"  Epoch {epoch}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Val RMSE: {val_metrics['rmse']:.4f} | Pearson: {val_metrics['pearson']:.4f}")
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"   -> New best model saved with Val RMSE: {best_val_rmse:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break
    elapsed = time.time() - t0

    model.load_state_dict(best_state)
    test_metrics, *_ = evaluate_model(model, test_loader, device, pkd_bounds)
    print(f"  Seed {seed} TEST: RMSE={test_metrics['rmse']:.4f} Pearson={test_metrics['pearson']:.4f} "
          f"Spearman={test_metrics['spearman']:.4f}  ({elapsed:.1f}s)")
    results.append({"seed": seed, **test_metrics, "train_time_s": elapsed})
    del model
    torch.cuda.empty_cache()

df = pd.DataFrame(results)
df.to_csv("abbind_fromscratch_results.csv", index=False)
print("\n=== SUMMARY (BALM-PPI from scratch, 81 AB-Bind complexes, 3 seeds) ===")
print(f"RMSE:    {df['rmse'].mean():.4f} +/- {df['rmse'].std():.4f}")
print(f"Pearson: {df['pearson'].mean():.4f} +/- {df['pearson'].std():.4f}")
print(f"Spearman:{df['spearman'].mean():.4f} +/- {df['spearman'].std():.4f}")
