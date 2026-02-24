import argparse
import os
from typing import List, Tuple, Optional
import torch
import numpy as np
import pandas as pd
import wandb
import esm
import gpytorch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr
from gpytorch.kernels import RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from balm import common_utils
from balm.models import BALM
from balm.models.utils import load_trained_model

def argument_parser():
    parser = argparse.ArgumentParser(
        description="Train Gaussian Process models on protein embeddings."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/protein_data.csv",
        help="Path to the dataset file.",
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        required=True,
        choices=["ESM2-protein", "ESM2-proteina", "BALM-protein", 
                "BALM-proteina", "BALM-concat", "BALM-sum", "BALM-subtract", "BALM-cosine"]
    )
    parser.add_argument("--test_size", type=float, default=0.8)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    return parser.parse_args()

def get_esm2_embeddings(sequences: List[str], batch_size: int = 32) -> np.ndarray:
    """Get ESM-2 embeddings for protein sequences"""
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().cuda()
    batch_converter = alphabet.get_batch_converter()
    
    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        data = [(str(j), seq) for j, seq in enumerate(batch_seqs)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.cuda()
        
        with torch.no_grad():
            results = model(tokens, repr_layers=[33])
            batch_embeddings = results["representations"][33].mean(axis=1)
            embeddings.append(batch_embeddings.cpu().numpy())
            
    return np.concatenate(embeddings)

def get_balm_embeddings(proteins_a: List[str], proteins_b: List[str], 
                       model: BALM, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get BALM embeddings for protein pairs"""
    protein_embeddings = []
    proteina_embeddings = []
    cosine_similarities = []
    
    for i in range(0, len(proteins_a), batch_size):
        batch_a = proteins_a[i:i + batch_size]
        batch_b = proteins_b[i:i + batch_size]
        
        inputs = {
            "protein_sequences": batch_a,
            "proteina_sequences": batch_b,
        }
        
        with torch.no_grad():
            outputs = model(inputs)
            protein_embeddings.append(outputs["protein_embedding"].cpu().numpy())
            proteina_embeddings.append(outputs["proteina_embedding"].cpu().numpy())
            cosine_similarities.append(outputs["cosine_similarity"].cpu().numpy())
    
    return (np.concatenate(protein_embeddings),
            np.concatenate(proteina_embeddings),
            np.concatenate(cosine_similarities))

class ProteinGPModel(gpytorch.models.ExactGP):
    """Gaussian Process model for protein embeddings"""
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_gp_model(X_train, y_train, X_test, y_test, sequences_test, 
                  learning_rate: float = 0.1, epochs: int = 50):
    """Train and evaluate GP model"""
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ProteinGPModel(X_train, y_train, likelihood)
    
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = ExactMarginalLogLikelihood(likelihood, model)
    
    for i in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
        wandb.log({"epoch": i+1, "train/loss": loss.item()})
    
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        preds = likelihood(model(X_test)).mean.numpy()
    
    metrics = {
        "rmse": np.sqrt(mean_squared_error(y_test, preds)),
        "r2": r2_score(y_test, preds),
        "spearman": spearmanr(y_test, preds)[0],
        "pearson": pearsonr(y_test, preds)[0]
    }
    
    wandb.log({f"test/{k}": v for k, v in metrics.items()})
    
    pd.DataFrame({
        "protein_a": [None] * len(y_test),
        "protein_b": sequences_test,
        "label": y_test,
        "prediction": preds
    }).to_csv("test_predictions.csv", index=False)
    
    return metrics

def main():
    args = argument_parser()
    data = load_dataset("BALM/BALM-benchmark", args.dataset, split="train").to_pandas()
    
    wandb.init(
        project="protein-protein-gp",
        name=f"gp_{args.embedding_type}_test{args.test_size}_lr{args.lr}",
        config=args
    )
    
    proteins_a = data["Target"].tolist()
    proteins_b = data["proteina"].tolist()
    y = data["Y"].to_numpy()
    
    if args.embedding_type.startswith("ESM2"):
        X = get_esm2_embeddings(proteins_a if "protein" in args.embedding_type else proteins_b)
    else:
        config_filepath = "default_configs/balm_peft.yaml"
        configs = Configs(**common_utils.load_yaml(config_filepath))
        model = BALM(configs.model_configs)
        model = load_trained_model(model, configs.model_configs, is_training=False)
        model.to("cuda")
        
        protein_emb, proteina_emb, cosine_sim = get_balm_embeddings(proteins_a, proteins_b, model)
        
        if args.embedding_type == "BALM-protein":
            X = protein_emb
        elif args.embedding_type == "BALM-proteina":
            X = proteina_emb
        elif args.embedding_type == "BALM-concat":
            X = np.concatenate((protein_emb, proteina_emb), axis=1)
        elif args.embedding_type == "BALM-sum":
            X = protein_emb + proteina_emb
        elif args.embedding_type == "BALM-subtract":
            X = protein_emb - proteina_emb
        else:  # BALM-cosine
            X = cosine_sim
    
    X_train, X_test, y_train, y_test, _, sequences_test = train_test_split(
        X, y, proteins_b, test_size=args.test_size, random_state=args.random_seed
    )
    
    torch.cuda.empty_cache()  # Clear memory before GP training
    
    metrics = train_gp_model(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
        sequences_test,
        args.lr,
        args.epochs
    )
    
    print(f"{args.embedding_type} GP Model Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    wandb.finish()

if __name__ == "__main__":
    main()