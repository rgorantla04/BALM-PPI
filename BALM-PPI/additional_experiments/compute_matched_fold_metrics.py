"""
Recompute PPB-Affinity baseline's per-fold Pearson/Spearman/RMSE using the
same methodology BALM-PPI's Table S2 uses (mean +/- s.d. of 5 SEPARATE
per-fold correlations), rather than PPB-Affinity's own "[All]" pooled-across-
all-folds correlation -- these are different statistics and shouldn't be
compared directly. We know exactly which fold each validation sample's PDB
belongs to from our own fold map, so we can reconstruct this.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

RESULTS_CSV = r"D:\BALM_Fineclone\external\PPB-Affinity\log_dir\baseline_train_config_2026_09_24__23_55_10_balm_cold_matched_nw0\checkpoints\results_100000.csv"
FOLD_MAP_PATH = r"D:\BALM_Fineclone\external\PPB-Affinity\balm_cold_fold_map.json"


def main():
    df = pd.read_csv(RESULTS_CSV)
    fold_map = json.load(open(FOLD_MAP_PATH))

    df["fold"] = df["complex"].map(fold_map)
    unmapped = df["fold"].isna().sum()
    print(f"Total validation rows: {len(df)}; unmapped (dropped from fold map): {unmapped}")
    df = df.dropna(subset=["fold"])
    df["fold"] = df["fold"].astype(int)

    fold_metrics = []
    for fold in sorted(df["fold"].unique()):
        d = df[df["fold"] == fold]
        rmse = np.sqrt(mean_squared_error(d["dG"], d["dG_pred"]))
        pearson = pearsonr(d["dG"], d["dG_pred"])[0]
        spearman = spearmanr(d["dG"], d["dG_pred"])[0]
        fold_metrics.append({"fold": fold, "n": len(d), "rmse": rmse, "pearson": pearson, "spearman": spearman})
        print(f"Fold {fold}: n={len(d)}  RMSE={rmse:.4f}  Pearson={pearson:.4f}  Spearman={spearman:.4f}")

    fm = pd.DataFrame(fold_metrics)
    print("\n=== PPB-Affinity baseline, per-fold mean +/- s.d. (matching BALM-PPI Table S2 methodology) ===")
    print(f"RMSE (dG, kcal/mol):  {fm['rmse'].mean():.4f} +/- {fm['rmse'].std():.4f}")
    print(f"Pearson:              {fm['pearson'].mean():.4f} +/- {fm['pearson'].std():.4f}")
    print(f"Spearman:             {fm['spearman'].mean():.4f} +/- {fm['spearman'].std():.4f}")

    # Also recompute the pooled statistic for reference / consistency check against the training log
    pooled_pearson = pearsonr(df["dG"], df["dG_pred"])[0]
    pooled_spearman = spearmanr(df["dG"], df["dG_pred"])[0]
    print(f"\n(Reference) Pooled-across-all-folds Pearson={pooled_pearson:.4f} Spearman={pooled_spearman:.4f}")

    fm.to_csv(r"D:\BALM_Fineclone\external\PPB-Affinity\matched_fold_metrics_final.csv", index=False)


if __name__ == "__main__":
    main()
