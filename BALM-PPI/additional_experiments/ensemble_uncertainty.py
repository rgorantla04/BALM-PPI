"""
Ensemble-based uncertainty estimate for BALM-PPI (Referee 2, point 7:
"no uncertainty/calibration estimation provided"), built from the 5
already-trained fold checkpoints of the random-split 5-fold CV run
(cv_results_lora_random_target/best_model_fold_{1..5}.pth) -- no retraining,
inference only.

Method: run all 5 independently-trained fold models on the full dataset,
treat the across-model standard deviation of the predicted pKd for each
sample as a per-prediction uncertainty proxy, and check whether that proxy
is actually informative by correlating it against the true absolute error
of the ensemble-mean prediction.

Honest scope limitation (to be stated explicitly in the rebuttal): each fold
model was trained on ~80% of the dataset (the other 4/5 of the random split),
so for most samples most of the 5 models HAVE seen that sample during their
own training. This is therefore a "prediction disagreement" uncertainty proxy
in the standard ensembling sense, not a strict epistemic/out-of-distribution
uncertainty measure -- we did not attempt to construct a fifth, entirely
held-out-by-all-5-models test set (e.g. from AB-Bind) within the scope of this
quick, low-cost analysis. If reviewers require the latter, it is a larger task
(needs a genuinely external, sequence-resolved held-out set) and would be
flagged as a follow-up rather than claimed here.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from train_multiseed_cv import (  # noqa: E402
    BALMForLoRAFinetuning,
    ProteinPairSequenceDataset,
    collate_fn,
    build_model,
)

CKPT_DIR = Path("BALM-PPI/scripts/cv_results_lora_random_target_new")  # HF-verified, matches Table S2
DATA_CSV = Path("BALM-PPI/scripts/notebooks/Data.csv")
N_FOLDS = 5
BATCH_SIZE = 4


@torch.no_grad()
def predict_all(model, loader, device, pkd_bounds):
    model.eval()
    preds = []
    lo, rng = pkd_bounds[0], pkd_bounds[1] - pkd_bounds[0]
    for batch in loader:
        out = model(batch)
        cos = out["cosine_similarity"].cpu()
        preds.append((((cos + 1) / 2) * rng + lo).numpy())
    return np.concatenate(preds)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(DATA_CSV)
    df = df.dropna(subset=["Target", "proteina", "Y", "PDB"]).reset_index(drop=True)
    df["Subgroup"] = df.get("Subgroup", "nan")
    pkd_bounds = (df["Y"].min(), df["Y"].max())
    y_true = df["Y"].to_numpy()

    dataset = ProteinPairSequenceDataset(df, pkd_bounds)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    cache_dir = Path("BALM-PPI/scripts/ensemble_uncertainty_fold_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_preds = []
    for fold_i in range(1, N_FOLDS + 1):
        cache_path = cache_dir / f"fold_{fold_i}_preds.npy"
        if cache_path.exists():
            preds = np.load(cache_path)
            all_preds.append(preds)
            print(f"  fold {fold_i} loaded from cache, {len(preds)} predictions")
            continue
        ckpt_path = CKPT_DIR / f"best_model_fold_{fold_i}.pth"
        print(f"Loading fold {fold_i}: {ckpt_path}")
        model = build_model(device, pkd_bounds)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        bad_missing = [k for k in missing if 'position_embeddings' not in k]
        bad_unexpected = [k for k in unexpected if 'position_embeddings' not in k]
        if bad_missing or bad_unexpected:
            raise RuntimeError(f"Unexpected state_dict mismatch -- missing={bad_missing}, unexpected={bad_unexpected}")
        preds = predict_all(model, loader, device, pkd_bounds)
        np.save(cache_path, preds)
        all_preds.append(preds)
        del model
        torch.cuda.empty_cache()
        print(f"  fold {fold_i} done, {len(preds)} predictions")

    all_preds = np.stack(all_preds, axis=0)  # (5, n_samples)
    ensemble_mean = all_preds.mean(axis=0)
    ensemble_std = all_preds.std(axis=0)
    abs_error = np.abs(ensemble_mean - y_true)

    print("\n=== Ensemble summary ===")
    print(f"n_samples = {len(y_true)}")
    print(f"Ensemble mean prediction RMSE: {np.sqrt(np.mean((ensemble_mean - y_true) ** 2)):.4f}")
    print(f"Mean ensemble std (uncertainty proxy): {ensemble_std.mean():.4f} +/- {ensemble_std.std():.4f}")

    r_pearson, p_pearson = pearsonr(ensemble_std, abs_error)
    r_spearman, p_spearman = spearmanr(ensemble_std, abs_error)
    print(f"\nCalibration check -- correlation between ensemble std (uncertainty) and |error|:")
    print(f"  Pearson  r = {r_pearson:.3f}  (p = {p_pearson:.2e})")
    print(f"  Spearman r = {r_spearman:.3f}  (p = {p_spearman:.2e})")

    # Bin by predicted uncertainty quartile and report mean absolute error per bin --
    # a well-calibrated uncertainty measure should show monotonically increasing
    # error with increasing predicted uncertainty.
    quartiles = pd.qcut(ensemble_std, 4, labels=["Q1 (lowest unc.)", "Q2", "Q3", "Q4 (highest unc.)"])
    calib_df = pd.DataFrame({"uncertainty_quartile": quartiles, "abs_error": abs_error})
    print("\nMean |error| by ensemble-uncertainty quartile:")
    print(calib_df.groupby("uncertainty_quartile", observed=True)["abs_error"].agg(["mean", "std", "count"]))

    out_df = pd.DataFrame(
        {
            "PDB": df["PDB"],
            "y_true": y_true,
            "ensemble_mean_pred": ensemble_mean,
            "ensemble_std": ensemble_std,
            "abs_error": abs_error,
        }
    )
    out_path = Path("BALM-PPI/scripts/ensemble_uncertainty_results.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved per-sample results to {out_path}")


if __name__ == "__main__":
    main()
