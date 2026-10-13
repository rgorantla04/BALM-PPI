"""
Retrain MVSF-AB's own published code (unmodified model/hyperparameters) on the
exact same de-overlapped 272-complex AB-Bind subset used for BALM-PPI's
zero/few-shot comparison (Data_cleaned.csv, reconstructed via
build_matched_split.py from the de-overlap logic: AB-Bind rows whose PDB
[first 4 chars of 'light'] overlaps with any PPB-Affinity PDB are removed --
272 of 1089 remain). Run with --train_csv/--test_csv/--seed pointing at the
real 30%/70% split files (mvsf_train_30pct_seed{42,123,999}.csv /
mvsf_test_30pct_seed{42,123,999}.csv) used in BALM-PPI's authoritative
few-shot notebook (AB-Bind_ALPINE_zero_few_shot.ipynb); the 0.4/1234
defaults below are legacy/unused and not what produced the reported numbers.

This mirrors the PPB-Affinity matched-fold rerun: their own code, their own
hyperparameters, only the train/test row assignment is substituted so the
comparison is apples-to-apples with BALM-PPI's own number on this exact data.

Two real bugs in the public repo were fixed to make this runnable at all:
1. `src/utils.py: seq_aaindex_dict` reads 'datasets/aaindex_pca.csv', which is
   not shipped in the repo (only aaindex1.csv is). Reconstructed via PCA(20)
   of aaindex1.csv (see build_matched_split.py) -- this is a reconstruction of
   a missing data asset, not a model/architecture change.
2. The default --seq-path/--feature-path (seq_natural.fasta /
   seq_natural_embedding.csv) only cover per-PDB wild-type sequences, not the
   per-mutant identifiers AB-Bind's pairs_abbind.csv actually uses. The
   correct pairing for AB-Bind (confirmed by row-count/order match) is
   datasets/seq.fasta + datasets/embedding.csv, which is what we use here.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent))
from src.models.mvsf import ModelAffinity
from src.utils import (
    PairedDataset,
    collate_paired_sequences,
    embed_dict,
    seq_aaindex_dict,
    SAM,
    log,
    pearsonr,
    r_squared,
    NormalizeData,
)


def r2_score(y_true, y_pred):
    return torch.tensor(r_squared(y_true, y_pred))

import argparse
_cli = argparse.ArgumentParser()
_cli.add_argument("--train_csv", default="mvsf_train_272.csv")
_cli.add_argument("--test_csv", default="mvsf_test_272.csv")
_cli.add_argument("--out_log", default="matched_split_train_log.txt")
_cli.add_argument("--out_pred_csv", default="matched_split_test_predictions.csv")
_cli.add_argument("--seed", type=int, default=1234)
_args, _ = _cli.parse_known_args()

TRAIN_CSV = _args.train_csv
TEST_CSV = _args.test_csv
SEQ_PATH = "datasets/seq.fasta"
FEATURE_PATH = "datasets/embedding.csv"
OUT_LOG = _args.out_log
OUT_PRED_CSV = _args.out_pred_csv

NUM_EPOCHS = 30
BATCH_SIZE = 16
LR = 1e-5
WEIGHT_DECAY = 1e-5
SEED = _args.seed

# fixed de-normalization constants from the original repo's model_eval(),
# unchanged -- these are dataset-wide min/max dG (kcal/mol) from the full
# AB-Bind/SKEMPI/SAbDab/Benchmark pooled training corpus, not fit to our subset.
MAX_VAL = 16.9138
MIN_VAL = 5.0400


def predict_affinity(model, Lchain, Hchain, antigen, embedding_tensor, aaindex_feature, use_cuda):
    b = len(Hchain)
    lchain_embeddings, hchain_embeddings, ag_embeddings = [], [], []
    lchain_aaindex, hchain_aaindex, ag_aaindex = [], [], []
    for i in range(b):
        lchain_embeddings.append(embedding_tensor[Lchain[i]])
        hchain_embeddings.append(embedding_tensor[Hchain[i]])
        ag_embeddings.append(embedding_tensor[antigen[i]])
        lchain_aaindex.append(aaindex_feature[Lchain[i]])
        hchain_aaindex.append(aaindex_feature[Hchain[i]])
        ag_aaindex.append(aaindex_feature[antigen[i]])

    lchain_embeddings = torch.stack(lchain_embeddings, 0)
    hchain_embeddings = torch.stack(hchain_embeddings, 0)
    ag_embeddings = torch.stack(ag_embeddings, 0)
    lchain_aaindex = torch.stack(lchain_aaindex, 0)
    hchain_aaindex = torch.stack(hchain_aaindex, 0)
    ag_aaindex = torch.stack(ag_aaindex, 0)
    if use_cuda:
        lchain_embeddings, hchain_embeddings, ag_embeddings = (
            lchain_embeddings.cuda(), hchain_embeddings.cuda(), ag_embeddings.cuda())
        lchain_aaindex, hchain_aaindex, ag_aaindex = (
            lchain_aaindex.cuda(), hchain_aaindex.cuda(), ag_aaindex.cuda())

    return model.predict(lchain_aaindex, hchain_aaindex, ag_aaindex,
                          lchain_embeddings, hchain_embeddings, ag_embeddings)


def evaluate(model, test_iterator, embeddings, aaindex_feature, use_cuda, save_preds=False):
    p_hat, true_y = [], []
    with torch.no_grad():
        for lchain, hchain, antigen, y in test_iterator:
            ph = predict_affinity(model, lchain, hchain, antigen, embeddings, aaindex_feature, use_cuda)
            p_hat.append(ph.detach().cpu().view(-1))
            true_y.append(y)
    y = torch.cat(true_y, 0).float()
    p_hat = torch.cat(p_hat, 0).float()

    p_hat_dg = (p_hat * (MAX_VAL - MIN_VAL)) + MIN_VAL  # de-normalize to raw dG scale
    rmse = torch.sqrt(torch.mean((y - p_hat_dg) ** 2)).item()
    mae = torch.mean(torch.abs(y - p_hat_dg)).item()
    r2 = r2_score(y, p_hat_dg).item()
    p = pearsonr(y, p_hat_dg)
    if save_preds:
        pd.DataFrame({"true_dG": y.numpy(), "pred_dG": p_hat_dg.numpy()}).to_csv(OUT_PRED_CSV, index=False)
    return rmse, mae, r2, p


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"Device: {device}")

    out = open(OUT_LOG, "w")

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    train_df.columns = ["light", "heavy", "antigen", "delta_g"]
    test_df.columns = ["light", "heavy", "antigen", "delta_g"]

    train_y = -torch.from_numpy(train_df["delta_g"].values)
    train_y = NormalizeData(train_y)
    # test_y uses the same -delta_g sign convention as train_y (unnormalized),
    # matching train.py's model_eval(), which compares raw -dG labels directly
    # against the de-normalized (but not sign-flipped) model output.
    test_y = -torch.from_numpy(test_df["delta_g"].values)

    train_dataset = PairedDataset(train_df["light"], train_df["heavy"], train_df["antigen"], train_y)
    train_iterator = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_paired_sequences,
        shuffle=True, pin_memory=False, drop_last=True)  # avoid BatchNorm1d crash on a size-1 final batch

    test_dataset = PairedDataset(test_df["light"], test_df["heavy"], test_df["antigen"], test_y)
    test_iterator = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_paired_sequences,
        shuffle=False, pin_memory=False, drop_last=False)

    all_proteins = (set(train_df["light"]) | set(train_df["heavy"]) | set(train_df["antigen"]) |
                     set(test_df["light"]) | set(test_df["heavy"]) | set(test_df["antigen"]))
    embeddings = embed_dict(SEQ_PATH, FEATURE_PATH)
    log(f"Loaded {len(embeddings)} sequence embeddings from {FEATURE_PATH}", file=out, print_also=True)
    aaindex_feature = seq_aaindex_dict(all_proteins, SEQ_PATH)
    log(f"Built aaindex features for {len(aaindex_feature)} proteins", file=out, print_also=True)

    model = ModelAffinity(BATCH_SIZE, use_cuda)
    model.use_cuda = use_cuda
    if use_cuda:
        model.cuda()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SAM(params, optim.Adam, lr=LR, weight_decay=WEIGHT_DECAY)

    log(f"Training MVSF-AB (unmodified architecture/hyperparameters) on matched 272-complex "
        f"de-overlapped AB-Bind set: train={len(train_df)} test={len(test_df)}", file=out, print_also=True)

    t0 = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        n, loss_accum = 0, 0.0
        optimizer.zero_grad()  # matches train.py's own loop: zeroed once per epoch, not per batch
        for lchain, hchain, antigen, y in train_iterator:
            phat = predict_affinity(model, lchain, hchain, antigen, embeddings, aaindex_feature, use_cuda)
            phat = phat.float().view(-1)
            if use_cuda:
                y = y.cuda()
            y = y.float()
            criterion = nn.MSELoss()
            loss = criterion(phat, y)
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            b = len(y)
            n += b
            loss_accum += (loss.item() - loss_accum) * b / n

        model.eval()
        rmse, mae, r2, p = evaluate(model, test_iterator, embeddings, aaindex_feature, use_cuda,
                                     save_preds=(epoch == NUM_EPOCHS - 1))
        elapsed = time.time() - t0
        log(f"Epoch {epoch+1}/{NUM_EPOCHS} [{elapsed:.0f}s] train_loss={loss_accum:.6f} "
            f"test_RMSE(dG)={rmse:.4f} test_MAE={mae:.4f} test_R2={r2:.4f} test_Pearson={p:.4f}",
            file=out, print_also=True)

    out.close()
    print("Done.")


if __name__ == "__main__":
    main()
