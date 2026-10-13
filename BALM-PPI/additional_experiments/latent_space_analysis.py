"""
Real latent-space characterization for Referee 2, point 1: does the 256-d
cosine latent space actually organize by binding energetics, using the
actual trained full BALM-PPI (LoRA) checkpoint on the cold split, fold 1
(results_multiseed_cold_balmppi/seed_42/best_model_fold_1.pth) -- the one
real cold-split LoRA checkpoint we have, evaluated on its own genuine held-out
test fold (never trained on by this checkpoint).

Two real analyses, no fabricated numbers:
  1. UMAP of the 256-d protein-chain projected embeddings for the fold-1 test
     set, colored by true pKd -- saved as a PNG for visual inspection.
  2. Pairwise latent cosine-distance vs. pairwise |delta pKd| correlation,
     computed ONLY within PDB groups (mutational series of the same complex)
     that have >=5 members in the fold-1 test set -- the correct way to test
     "does distance in latent space track affinity change" without conflating
     it with across-complex structural differences.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
PROJECTED_SIZE = 256
DROPOUT = 0.1
LORA_RANK, LORA_ALPHA, LORA_DROPOUT = 8, 16, 0.1
DATA_CSV = r"D:\BALM_Fineclone\BALM-PPI\scripts\notebooks\Data.csv"
# CORRECTED: the previous checkpoint (results_multiseed_cold_balmppi/...) was from
# an abandoned local rerun and does NOT match the published model (fold-1 Pearson
# 0.707 vs the real 0.765). This is the authoritative checkpoint downloaded from
# huggingface.co/Harshit494/ALPINE/cv_results_lora/best_model_fold_1.pth, verified
# to reproduce the real published per-fold number.
CHECKPOINT = r"D:\BALM_Fineclone\BALM-PPI\scripts\real_checkpoint\best_model_fold_1.pth"


class BALMProjectionHead(nn.Module):
    def __init__(self, embedding_size, projected_size, dropout):
        super().__init__()
        self.protein_projection = nn.Linear(embedding_size, projected_size)
        self.proteina_projection = nn.Linear(embedding_size, projected_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, protein_embedding, proteina_embedding):
        protein_embedding = self.dropout(protein_embedding)
        proteina_embedding = self.dropout(proteina_embedding)
        p = F.normalize(self.protein_projection(protein_embedding), p=2, dim=1)
        q = F.normalize(self.proteina_projection(proteina_embedding), p=2, dim=1)
        return p, q


class BALMForLoRAFinetuning(nn.Module):
    def __init__(self, esm_model, esm_tokenizer, projected_size, dropout):
        super().__init__()
        self.esm_model = esm_model
        self.esm_tokenizer = esm_tokenizer
        embedding_size = esm_model.config.hidden_size
        self.projection_head = BALMProjectionHead(embedding_size, projected_size, dropout)
        self.cls_token = esm_tokenizer.cls_token

    def _get_esm_embeddings(self, sequences):
        processed = [s.replace("|", f"{self.cls_token}{self.cls_token}") for s in sequences]
        inputs = self.esm_tokenizer(processed, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(self.esm_model.device) for k, v in inputs.items()}
        outputs = self.esm_model(**inputs)
        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    @torch.no_grad()
    def get_embeddings(self, protein_seqs, proteina_seqs, batch_size=8):
        ps, qs = [], []
        for i in range(0, len(protein_seqs), batch_size):
            p_emb = self._get_esm_embeddings(protein_seqs[i:i + batch_size])
            q_emb = self._get_esm_embeddings(proteina_seqs[i:i + batch_size])
            p, q = self.projection_head(p_emb, q_emb)
            ps.append(p.cpu())
            qs.append(q.cpu())
        return torch.cat(ps), torch.cat(qs)


REAL_FOLD1_CSV = r"D:\BALM_Fineclone\BALM-PPI\scripts\cv_results_lora_full_cold\fold_1_predictions.csv"


def main():
    from peft import LoraConfig, get_peft_model, TaskType

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(DATA_CSV)
    df = df.dropna(subset=["Target", "proteina", "Y", "PDB"]).reset_index(drop=True)

    # Use the REAL fold-1 partition, taken directly from the authoritative
    # HuggingFace predictions file -- NOT a local GroupKFold reconstruction,
    # which was found to diverge substantially (only 115/607 PDBs overlapping
    # with the real fold-1 PDB set) from the actual training pipeline's split.
    real_fold1 = pd.read_csv(REAL_FOLD1_CSV)
    real_test_pdbs = set(real_fold1["PDB"].unique())
    print(f"Real fold-1 test PDBs (from HuggingFace file): {len(real_test_pdbs)}")

    is_test = df["PDB"].isin(real_test_pdbs)
    test_df = df[is_test].reset_index(drop=True)
    train_df_full = df[~is_test].reset_index(drop=True)
    print(f"Fold-1 test set: {len(test_df)} rows, {test_df['PDB'].nunique()} unique PDBs")
    print(f"Reconstructed train set: {len(train_df_full)} rows, {train_df_full['PDB'].nunique()} unique PDBs")

    # Verify row count matches the real file closely (should be exact or very close)
    if len(test_df) != len(real_fold1):
        print(f"WARNING: row count mismatch -- local test_df={len(test_df)} vs real file={len(real_fold1)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["key", "query", "value"],
    )
    base = get_peft_model(base, lora_config)
    model = BALMForLoRAFinetuning(base, tokenizer, PROJECTED_SIZE, DROPOUT).to(device)

    state = torch.load(CHECKPOINT, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint. Missing keys: {len(missing)}, unexpected: {len(unexpected)}")
    if len(missing) > 5:
        print("WARNING: many missing keys, checkpoint structure may not match:", missing[:10])
    model.eval()

    # Sanity check against the known-real per-fold number before trusting anything downstream.
    with torch.no_grad():
        check_preds = []
        for i in range(0, len(test_df), 8):
            batch = test_df.iloc[i:i+8]
            p, q = model.get_embeddings(list(batch["Target"]), list(batch["proteina"]), batch_size=8)
            p_n, q_n = F.normalize(p, dim=1), F.normalize(q, dim=1)
            cos = (p_n * q_n).sum(dim=1)
            check_preds.extend(cos.tolist())
    from scipy.stats import pearsonr as _pr
    y_min, y_max = df["Y"].min(), df["Y"].max()
    check_pkd = [((c + 1) / 2) * (y_max - y_min) + y_min for c in check_preds]
    r_check = _pr(test_df["Y"], check_pkd)[0]
    print(f"SANITY CHECK -- recomputed fold-1 Pearson from this checkpoint: {r_check:.4f} (expected ~0.765)")

    p_emb, q_emb = model.get_embeddings(list(test_df["Target"]), list(test_df["proteina"]), batch_size=8)
    print("Extracted TEST embeddings:", p_emb.shape, q_emb.shape)
    joint = torch.cat([p_emb, q_emb], dim=1).numpy()
    np.save("latent_joint_embeddings_fold1_REAL.npy", joint)
    test_df.to_csv("latent_embeddings_fold1_meta_REAL.csv", index=False)

    # Also extract TRAIN-partition embeddings (needed for the latent-distance-to-
    # training-manifold uncertainty experiment) using the same real checkpoint.
    train_df = train_df_full
    print(f"Extracting TRAIN embeddings for {len(train_df)} rows (this will take a while)...")
    p_emb_tr, q_emb_tr = model.get_embeddings(list(train_df["Target"]), list(train_df["proteina"]), batch_size=8)
    joint_tr = torch.cat([p_emb_tr, q_emb_tr], dim=1).numpy()
    np.save("latent_joint_embeddings_fold1_TRAIN_REAL.npy", joint_tr)
    train_df.to_csv("latent_embeddings_fold1_TRAIN_meta_REAL.csv", index=False)
    print("Extracted TRAIN embeddings:", joint_tr.shape)

    # --- UMAP ---
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        coords = reducer.fit_transform(joint)
        np.save("latent_umap_coords_fold1_REAL.npy", coords)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 6))
        sc = plt.scatter(coords[:, 0], coords[:, 1], c=test_df["Y"], cmap="viridis", s=8, alpha=0.7)
        plt.colorbar(sc, label="True pKd")
        plt.title("BALM-PPI latent space (fold-1 cold-split test set, real checkpoint)")
        plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
        plt.tight_layout()
        plt.savefig("latent_umap_fold1_REAL.png", dpi=150)
        print("Saved UMAP plot to latent_umap_fold1_REAL.png")
    except ImportError:
        print("umap-learn not installed, skipping UMAP plot (pairwise analysis below is independent of this).")

    # --- Pairwise latent distance vs |delta pKd|, within PDB mutational series ---
    all_true_dist, all_true_dy = [], []
    per_pdb_results = []
    for pdb, g in test_df.groupby("PDB"):
        if len(g) < 5:
            continue
        idx = g.index.to_numpy()
        sub_joint = joint[idx]
        y = g["Y"].to_numpy()
        n = len(idx)
        dists, dys = [], []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(sub_joint[i] - sub_joint[j])  # euclidean in joint projected space
                dy = abs(y[i] - y[j])
                dists.append(d)
                dys.append(dy)
        if len(dists) >= 10:
            r, p = pearsonr(dists, dys)
            per_pdb_results.append({"PDB": pdb, "n_variants": n, "n_pairs": len(dists), "pearson_dist_vs_dy": r, "p_value": p})
            all_true_dist.extend(dists)
            all_true_dy.extend(dys)

    res_df = pd.DataFrame(per_pdb_results).sort_values("n_pairs", ascending=False)
    res_df.to_csv("latent_distance_vs_affinity_per_pdb_REAL.csv", index=False)
    print(res_df.to_string())

    overall_r, overall_p = pearsonr(all_true_dist, all_true_dy)
    print(f"\n=== POOLED across all qualifying PDB mutational series ===")
    print(f"n_pdbs={len(per_pdb_results)}, n_pairs={len(all_true_dist)}")
    print(f"Pearson(pairwise latent distance, pairwise |delta pKd|) = {overall_r:.4f} (p={overall_p:.2e})")


if __name__ == "__main__":
    main()
