"""
Train Ridge / Random Forest / XGBoost regression heads on frozen MINT pair
embeddings, under the EXACT SAME random / cold_target / sequence_similarity
5-fold splits used for BALM-PPI and the existing classical baselines
(classical_baselines.py), for a genuinely matched comparison against a recent
sequence-based PPI-pretrained method (Referee 3 point 4).

This is fine-tuned/trained on OUR data under OUR folds (unlike PPLM-Affinity,
which was evaluated zero-shot) -- MINT's own backbone is frozen (matching its
own established downstream-task methodology, see downstream/GeneralPPI/
finetune_general.py's Ridge/MLP-on-frozen-embeddings approach), but the
regression head is trained from scratch on PPB-Affinity under our exact CV
protocol, making it directly comparable to BALM-PPI, BALM-PPI-without-PEFT,
the concatenation baseline, and the RF/XGBoost-on-ESM-2 classical baselines.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold

DATA_CSV = r"D:\BALM_Fineclone\BALM-PPI\scripts\notebooks\Data.csv"
EMB_PATH = r"D:\BALM_Fineclone\external\mint\ppb_affinity_mint_embeddings.pt"
N_SPLITS = 5
SEED = 42
K_MER = 3
SIM_THRESHOLD = 0.3


def get_random_folds(n_rows):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(kf.split(np.arange(n_rows)))


def get_cold_folds(df):
    gkf = GroupKFold(n_splits=N_SPLITS)
    groups = df["PDB"].factorize()[0]
    return list(gkf.split(df, groups=groups))


def get_sequence_similarity_folds(df):
    all_sequences = sorted(set(df["Target"]) | set(df["proteina"]))
    seq_to_idx = {s: i for i, s in enumerate(all_sequences)}

    def get_kmers(seq):
        if len(seq) < K_MER:
            return {seq}
        return {seq[i : i + K_MER] for i in range(len(seq) - K_MER + 1)}

    kmer_sets = [get_kmers(s) for s in all_sequences]
    n = len(all_sequences)
    sim = np.zeros((n, n))
    for i in range(n):
        sim[i, i] = 1.0
        for j in range(i + 1, n):
            inter = len(kmer_sets[i] & kmer_sets[j])
            union = len(kmer_sets[i] | kmer_sets[j])
            s = inter / union if union > 0 else 0.0
            sim[i, j] = sim[j, i] = s
    dist = 1.0 - sim
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    clusters = fcluster(Z, t=1.0 - SIM_THRESHOLD, criterion="distance") - 1
    cluster_ids = sorted(set(clusters))

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    seq_cluster = {s: clusters[seq_to_idx[s]] for s in all_sequences}

    folds = []
    for train_c_idx, test_c_idx in kf.split(cluster_ids):
        train_clusters = {cluster_ids[i] for i in train_c_idx}
        test_clusters = {cluster_ids[i] for i in test_c_idx}
        seq_group = {s: ("test" if c in test_clusters else "train") for s, c in seq_cluster.items()}
        train_idx, test_idx = [], []
        for idx, row in df.iterrows():
            if seq_group.get(row["Target"]) == "test" or seq_group.get(row["proteina"]) == "test":
                test_idx.append(idx)
            else:
                train_idx.append(idx)
        folds.append((np.array(train_idx), np.array(test_idx)))
    return folds


def concordance_index(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    if n < 2:
        return 1.0
    concordant = discordant = tied = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            if y_pred[i] == y_pred[j]:
                tied += 1
            elif (y_pred[i] > y_pred[j]) == (y_true[i] > y_true[j]):
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant + tied
    return 1.0 if total == 0 else (concordant + 0.5 * tied) / total


def evaluate_fold(model_ctor, X_train, y_train, X_test, y_test):
    model = model_ctor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "pearson": float(pearsonr(y_test, preds)[0]),
        "spearman": float(spearmanr(y_test, preds)[0]),
        "ci": float(concordance_index(y_test, preds)),
    }


def run_all(X, y, folds, split_name, results):
    model_ctors = {
        "Ridge": lambda: Ridge(alpha=1.0, random_state=SEED),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED),
        "XGBoost": lambda: xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05, n_jobs=-1, random_state=SEED, tree_method="hist"
        ),
    }
    for model_name, ctor in model_ctors.items():
        fold_metrics = []
        for fold_i, (train_idx, test_idx) in enumerate(folds, start=1):
            m = evaluate_fold(ctor, X[train_idx], y[train_idx], X[test_idx], y[test_idx])
            print(f"[{split_name}][{model_name}] fold {fold_i}: {m}")
            fold_metrics.append(m)
        df_m = pd.DataFrame(fold_metrics)
        summary = {f"{k}_mean": df_m[k].mean() for k in df_m.columns}
        summary.update({f"{k}_std": df_m[k].std() for k in df_m.columns})
        summary["split"] = split_name
        summary["model"] = f"MINT-{model_name}"
        results.append(summary)


def main():
    print("Loading data and MINT embeddings...")
    df = pd.read_csv(DATA_CSV).dropna(subset=["Target", "proteina", "Y", "PDB"]).reset_index(drop=True)
    emb_data = torch.load(EMB_PATH)
    X = emb_data["embeddings"].numpy()
    y = df["Y"].to_numpy(dtype=np.float32)
    assert len(X) == len(df), f"Embedding count {len(X)} != dataframe rows {len(df)}"
    print(f"Dataset: {len(df)} rows, embedding dim {X.shape[1]}")

    results = []
    print("\n=== RANDOM SPLIT ===")
    run_all(X, y, get_random_folds(len(df)), "random", results)
    print("\n=== COLD_TARGET SPLIT ===")
    run_all(X, y, get_cold_folds(df), "cold_target", results)
    print("\n=== SEQUENCE_SIMILARITY SPLIT ===")
    run_all(X, y, get_sequence_similarity_folds(df), "sequence_similarity", results)

    out = pd.DataFrame(results)
    out_path = Path(r"D:\BALM_Fineclone\external\mint\mint_regression_results.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(out[["split", "model", "rmse_mean", "rmse_std", "pearson_mean", "pearson_std", "spearman_mean", "spearman_std"]])


if __name__ == "__main__":
    main()
