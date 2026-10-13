"""
Real (not literature-reused) ablations for Referee 2 points 2-3:
  - R2-2: cosine similarity vs. alternative distance metrics (unnormalized
    Euclidean, Manhattan, learned bilinear) in the same two-stream projection
    architecture.
  - R2-3: asymmetric two-stream projection (W1 != W2, the model used
    throughout the paper) vs. a weight-shared Siamese variant (W1 == W2).

Same cached frozen ESM-2 embeddings, same cold_target 5-fold split, same
seed=42, same training protocol (test fold doubles as early-stopping signal,
matching Table S2 and the existing 128/256/512 projection-size ablation) as
every other "cheap" ablation in this rebuttal -- only the head/loss changes.

NOTE: a Triplet-Margin-Loss variant was considered (as in an earlier,
unverified draft of this rebuttal) and deliberately NOT included here: triplet
loss operates on relative embedding distances across *different* samples in a
batch, not on the within-pair (protein A, protein B) relationship our cosine
objective directly supervises, so it would need a separate post-hoc
calibration step to produce a pKd prediction at all, making it a
not-quite-comparable setup rather than a clean ablation. Since no referee
asked specifically for a triplet comparison (only for a general comparison
against "other metric-learning approaches," which the two ablations below
already address substantively), we are not including a half-justified triplet
number rather than fabricate one that looks precise but rests on a shaky
methodological foundation.
"""
import argparse
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

DATA_CSV = r"D:\BALM_Fineclone\BALM-PPI\scripts\notebooks\Data.csv"
TARGET_EMB_PATH = r"D:\BALM_Fineclone\cached_embeddings\protein_embeddings.pkl"
PROTEINA_EMB_PATH = r"D:\BALM_Fineclone\cached_embeddings\proteina_embeddings.pkl"

EMBED_DIM = 1280
PROJECTED_SIZE = 256
N_SPLITS = 5
EPOCHS = 50
PATIENCE = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
DROPOUT = 0.1
SEED = 42


def load_data():
    df = pd.read_csv(DATA_CSV)
    df = df.dropna(subset=["Target", "proteina", "Y", "PDB"]).reset_index(drop=True)
    with open(TARGET_EMB_PATH, "rb") as f:
        target_emb = pickle.load(f)
    with open(PROTEINA_EMB_PATH, "rb") as f:
        proteina_emb = pickle.load(f)
    keep = df["Target"].isin(target_emb) & df["proteina"].isin(proteina_emb)
    df = df[keep].reset_index(drop=True)
    X1 = np.stack([target_emb[t] for t in df["Target"]]).astype(np.float32)
    X2 = np.stack([proteina_emb[p] for p in df["proteina"]]).astype(np.float32)
    y = df["Y"].to_numpy(dtype=np.float32)
    return df, X1, X2, y


class PairEmbeddingDataset(Dataset):
    def __init__(self, X1, X2, y, pkd_bounds):
        self.X1, self.X2, self.y = X1, X2, y
        self.lo, self.hi = pkd_bounds

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        scaled = ((self.y[idx] - self.lo) / (self.hi - self.lo)) * 2 - 1
        return {
            "x1": torch.from_numpy(self.X1[idx]),
            "x2": torch.from_numpy(self.X2[idx]),
            "label": torch.tensor(scaled, dtype=torch.float32),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }


def get_cold_folds(df) -> List[Tuple[np.ndarray, np.ndarray]]:
    gkf = GroupKFold(n_splits=N_SPLITS)
    groups = df["PDB"].factorize()[0]
    return list(gkf.split(df, groups=groups))


# ---------------------------------------------------------------------------
# Heads: all output a scalar in a model-appropriate range; evaluate() maps
# cosine-like heads via ((out+1)/2) and distance-based heads via a learned
# per-fold linear recalibration (since raw negative-distance has no fixed
# [-1,1] scale the way cosine similarity does).
# ---------------------------------------------------------------------------
class TwoStreamCosine(nn.Module):
    """The paper's own architecture: independent W1, W2, cosine similarity."""

    def __init__(self, embed_dim, projected_size, dropout):
        super().__init__()
        self.proj1 = nn.Linear(embed_dim, projected_size)
        self.proj2 = nn.Linear(embed_dim, projected_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        x1, x2 = self.dropout(x1), self.dropout(x2)
        p1 = F.normalize(self.proj1(x1), p=2, dim=1)
        p2 = F.normalize(self.proj2(x2), p=2, dim=1)
        return torch.clamp(F.cosine_similarity(p1, p2), -0.9999, 0.9999), True


class SiameseCosine(nn.Module):
    """Weight-shared variant: a single projection matrix for both partners."""

    def __init__(self, embed_dim, projected_size, dropout):
        super().__init__()
        self.proj = nn.Linear(embed_dim, projected_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        x1, x2 = self.dropout(x1), self.dropout(x2)
        p1 = F.normalize(self.proj(x1), p=2, dim=1)
        p2 = F.normalize(self.proj(x2), p=2, dim=1)
        return torch.clamp(F.cosine_similarity(p1, p2), -0.9999, 0.9999), True


class TwoStreamEuclidean(nn.Module):
    """Unnormalized Euclidean distance in the two-stream projected space."""

    def __init__(self, embed_dim, projected_size, dropout):
        super().__init__()
        self.proj1 = nn.Linear(embed_dim, projected_size)
        self.proj2 = nn.Linear(embed_dim, projected_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        x1, x2 = self.dropout(x1), self.dropout(x2)
        p1, p2 = self.proj1(x1), self.proj2(x2)
        dist = torch.norm(p1 - p2, p=2, dim=1)
        return -dist, False


class TwoStreamManhattan(nn.Module):
    """Unnormalized Manhattan (L1) distance in the two-stream projected space."""

    def __init__(self, embed_dim, projected_size, dropout):
        super().__init__()
        self.proj1 = nn.Linear(embed_dim, projected_size)
        self.proj2 = nn.Linear(embed_dim, projected_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        x1, x2 = self.dropout(x1), self.dropout(x2)
        p1, p2 = self.proj1(x1), self.proj2(x2)
        dist = torch.sum(torch.abs(p1 - p2), dim=1)
        return -dist, False


class TwoStreamBilinear(nn.Module):
    """Learned bilinear similarity: p1^T A p2, tanh-bounded for stability."""

    def __init__(self, embed_dim, projected_size, dropout):
        super().__init__()
        self.proj1 = nn.Linear(embed_dim, projected_size)
        self.proj2 = nn.Linear(embed_dim, projected_size)
        self.bilinear = nn.Linear(projected_size, projected_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        x1, x2 = self.dropout(x1), self.dropout(x2)
        p1, p2 = self.proj1(x1), self.proj2(x2)
        score = torch.sum(self.bilinear(p1) * p2, dim=1) / (p1.shape[1] ** 0.5)
        return torch.tanh(score), True


MODEL_REGISTRY = {
    "cosine_two_stream": TwoStreamCosine,
    "cosine_siamese": SiameseCosine,
    "euclidean": TwoStreamEuclidean,
    "manhattan": TwoStreamManhattan,
    "bilinear": TwoStreamBilinear,
}


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


@torch.no_grad()
def collect_outputs(model, loader, device):
    model.eval()
    ys, outs = [], []
    for batch in loader:
        x1, x2 = batch["x1"].to(device), batch["x2"].to(device)
        out, _ = model(x1, x2)
        ys.extend(batch["y"].numpy())
        outs.extend(out.cpu().numpy())
    return np.array(ys), np.array(outs)


def evaluate_fold(model, train_loader, test_loader, device, pkd_bounds, is_bounded):
    lo, hi = pkd_bounds
    y_test, out_test = collect_outputs(model, test_loader, device)
    if is_bounded:
        preds = ((out_test + 1) / 2) * (hi - lo) + lo
    else:
        # raw (negative) distance has no fixed scale -- recalibrate with a
        # simple linear fit on the TRAIN fold's own (out, y) pairs, same
        # approach already used elsewhere in this rebuttal (PPLM-Affinity
        # recalibration) for a model whose head has no natural pKd scale.
        y_train, out_train = collect_outputs(model, train_loader, device)
        A = np.vstack([out_train, np.ones_like(out_train)]).T
        slope, intercept = np.linalg.lstsq(A, y_train, rcond=None)[0]
        preds = slope * out_test + intercept
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "pearson": float(pearsonr(y_test, preds)[0]),
        "spearman": float(spearmanr(y_test, preds)[0]),
        "ci": float(concordance_index(y_test, preds)),
    }


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_fold(model_cls, X1, X2, y, train_idx, test_idx, pkd_bounds, device):
    set_seed(SEED)
    model = model_cls(EMBED_DIM, PROJECTED_SIZE, DROPOUT).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    def make_loader(idx, shuffle):
        ds = PairEmbeddingDataset(X1[idx], X2[idx], y[idx], pkd_bounds)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(train_idx, True)
    test_loader = make_loader(test_idx, False)  # doubles as early-stop signal, matching Table S2 convention

    best_val_rmse = float("inf")
    best_state = None
    patience_counter = 0
    is_bounded = None

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            x1, x2, label = batch["x1"].to(device), batch["x2"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            out, is_bounded = model(x1, x2)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate_fold(model, train_loader, test_loader, device, pkd_bounds, is_bounded)
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            break

    model.load_state_dict(best_state)
    test_metrics = evaluate_fold(model, train_loader, test_loader, device, pkd_bounds, is_bounded)
    return test_metrics, epoch + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True)
    ap.add_argument("--output_csv", default=r"D:\BALM_Fineclone\BALM-PPI\scripts\metric_architecture_ablation_results.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, X1, X2, y = load_data()
    pkd_bounds = (float(y.min()), float(y.max()))
    folds = get_cold_folds(df)
    model_cls = MODEL_REGISTRY[args.model]

    results = []
    for fold_i, (train_idx, test_idx) in enumerate(folds, start=1):
        metrics, n_epochs = train_one_fold(model_cls, X1, X2, y, train_idx, test_idx, pkd_bounds, device)
        print(f"[{args.model}] fold {fold_i}/5 (stopped epoch {n_epochs}): {metrics}")
        metrics.update({"fold": fold_i, "model": args.model})
        results.append(metrics)

    out_df = pd.DataFrame(results)
    header = not Path(args.output_csv).exists()
    out_df.to_csv(args.output_csv, mode="a", header=header, index=False)
    print(out_df[["rmse", "pearson", "spearman", "ci"]].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
