"""
Full breakdown of the retrained PPB-Affinity baseline (identical cold-split
folds to BALM-PPI) by subgroup (TCR-pMHC, Antibody-Antigen, Overall) and by
source dataset (SKEMPI v2.0, SAbDab, Affinity Benchmark v5.5, PDBbind v2020,
ATLAS) -- matching the granularity of the original manuscript's Table S3/S4,
now computed under a genuinely matched-fold comparison.

Also reports RMSE in BOTH native dG units (kcal/mol, PPB-Affinity's training
objective) and a pKd-equivalent unit via the thermodynamic relation
dG = -RT*ln(Ka) = RT*ln(Kd) = -RT*ln(10)*pKd, i.e. 1 pKd unit ~= RT*ln(10)
= 1.364 kcal/mol at T=298.15K, so RMSE_pKd_equivalent = RMSE_dG / 1.364.
This is NOT the same as the per-fold linear recalibration used for PPLM
(there, the model's raw scale was arbitrary; here, dG and pKd are related by
a known, fixed physical constant, not a fitted slope) -- so this conversion
is reported as a direct unit change, appropriate for comparing directly
against BALM-PPI's own pKd-scale RMSE without any fitting.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

RESULTS_CSV = r"D:\BALM_Fineclone\external\PPB-Affinity\log_dir\baseline_train_config_2026_09_24__23_55_10_balm_cold_matched_nw0\checkpoints\results_100000.csv"
BENCHMARK_CSV = r"D:\BALM_Fineclone\external\PPB-Affinity\benchmark.csv"
FOLD_MAP_PATH = r"D:\BALM_Fineclone\external\PPB-Affinity\balm_cold_fold_map.json"
RT_LN10 = 1.364  # kcal/mol per pKd unit at 298.15K


def metrics_for(d):
    rmse_dg = float(np.sqrt(mean_squared_error(d["dG"], d["dG_pred"])))
    pearson = float(pearsonr(d["dG"], d["dG_pred"])[0])
    spearman = float(spearmanr(d["dG"], d["dG_pred"])[0])
    return {
        "n": len(d),
        "rmse_dG_kcal_mol": rmse_dg,
        "rmse_pKd_equivalent": rmse_dg / RT_LN10,
        "pearson": pearson,
        "spearman": spearman,
    }


def main():
    df = pd.read_csv(RESULTS_CSV)
    bench = pd.read_csv(BENCHMARK_CSV)
    fold_map = json.load(open(FOLD_MAP_PATH))

    # Map each result row's "complex" (PDB code) to Subgroup and source via benchmark.csv
    pdb_to_subgroup = bench.drop_duplicates("pdb").set_index("pdb")["Subgroup"].to_dict()
    pdb_to_source = bench.drop_duplicates("pdb").set_index("pdb")["source"].to_dict()

    df["fold"] = df["complex"].map(fold_map)
    df = df.dropna(subset=["fold"]).copy()
    df["Subgroup"] = df["complex"].map(pdb_to_subgroup)
    df["source"] = df["complex"].map(pdb_to_source)

    print("=" * 80)
    print("OVERALL (per-fold mean +/- s.d., n=5 folds) -- for reference / sanity check")
    print("=" * 80)
    fold_metrics = [metrics_for(df[df["fold"] == f]) for f in sorted(df["fold"].unique())]
    fm = pd.DataFrame(fold_metrics)
    for col in ["rmse_dG_kcal_mol", "rmse_pKd_equivalent", "pearson", "spearman"]:
        print(f"  {col}: {fm[col].mean():.4f} +/- {fm[col].std():.4f}")

    print()
    print("=" * 80)
    print("BY SUBGROUP (pooled across all folds -- single fixed model per fold, evaluated on subgroup subset)")
    print("=" * 80)
    for subgroup in ["TCR-pMHC", "Antibody-Antigen"]:
        sub = df[df["Subgroup"] == subgroup]
        if len(sub) == 0:
            print(f"  {subgroup}: NO DATA in matched-fold population")
            continue
        m = metrics_for(sub)
        print(f"  {subgroup} (n={m['n']}): RMSE(dG)={m['rmse_dG_kcal_mol']:.3f} kcal/mol  "
              f"RMSE(pKd-equiv)={m['rmse_pKd_equivalent']:.3f}  Pearson={m['pearson']:.3f}  Spearman={m['spearman']:.3f}")
    overall = df[df["Subgroup"].isna()]
    m = metrics_for(df)
    print(f"  Entire (n={m['n']}): RMSE(dG)={m['rmse_dG_kcal_mol']:.3f} kcal/mol  "
          f"RMSE(pKd-equiv)={m['rmse_pKd_equivalent']:.3f}  Pearson={m['pearson']:.3f}  Spearman={m['spearman']:.3f}")

    print()
    print("=" * 80)
    print("BY SOURCE DATASET (pooled across all folds)")
    print("=" * 80)
    for source in ["SKEMPI v2.0", "SAbDab", "Affinity Benchmark v5.5", "PDBbind v2020", "ATLAS"]:
        sub = df[df["source"] == source]
        if len(sub) == 0:
            print(f"  {source}: NO DATA in matched-fold population")
            continue
        m = metrics_for(sub)
        print(f"  {source} (n={m['n']}): RMSE(dG)={m['rmse_dG_kcal_mol']:.3f} kcal/mol  "
              f"RMSE(pKd-equiv)={m['rmse_pKd_equivalent']:.3f}  Pearson={m['pearson']:.3f}  Spearman={m['spearman']:.3f}")

    df.to_csv(r"D:\BALM_Fineclone\external\PPB-Affinity\matched_results_with_subgroup_source.csv", index=False)


if __name__ == "__main__":
    main()
