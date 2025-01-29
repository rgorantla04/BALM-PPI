import argparse
import os
import sys
import torch
import esm
import numpy as np
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
sys.path.append(os.getcwd())

from dotenv import load_dotenv
import wandb
import pandas as pd
import numpy as np
import torch
import gpytorch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
from gpytorch.kernels import RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from balm import common_utils
from balm.models.utils import load_trained_model, load_pretrained_pkd_bounds
from balm.configs import Configs
from balm.models import BALM
import gpytorch
from gpytorch.kernels import RBFKernel


def argument_parser():
    """
    Parse the command line arguments for the script.

    Returns:
        argparse.Namespace: Parsed command-line arguments with options such as dataset path, embedding type,
                            test size, learning rate, and number of epochs.
    """
    parser = argparse.ArgumentParser(
        description="Train Gaussian Process models on ECFP8 and ChemBERTa embeddings."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/Mpro.csv",
        help="Path to the dataset file.",
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        required=True,
choices=["ESM2", "ESM2-proteina", "BALM-protein", "BALM-proteina", "BALM-concat", "BALM-sum", "BALM-subtract", "BALM-cosine"]        help="Type of embedding on which the GP is trained on",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.8, help="The ratio of the test set."
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for data splitting."
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training iterations."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run the script in debug mode."
    )

    return parser.parse_args()



def smiles_to_ecfp8_fingerprint(smiles_list):
    """
    Convert protein sequences to ESM-2 embeddings.

    Args:
        smiles_list (list of str): A list of protein sequences.

    Returns:
        np.ndarray: A NumPy array containing ESM-2 embeddings for each protein sequence.
    """
    fingerprints = []
    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().cuda()  # Move to GPU if available
    batch_converter = alphabet.get_batch_converter()
    
    for smi in smiles_list:
        try:
            # Prepare data in the format expected by ESM-2
            data = [(str(i), smi) for i, smi in enumerate([smi])]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.cuda()  # Move to GPU if available
            
            # Generate embeddings
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33])
            embedding = results["representations"][33].cpu().numpy()
            
            # Use mean pooling over sequence length to get fixed-size representation
            sequence_embedding = embedding.mean(axis=1)
            fingerprints.append(sequence_embedding[0])  # Take first (and only) sequence
            
        except Exception as e:
            # Handle invalid sequences with zeros
            fingerprints.append(np.zeros(1280))  # ESM-2 has 1280 dimensions
            
    return np.array(fingerprints)

def smiles_to_chemberta_embedding(smiles_list, model, batch_converter):
    """
    Convert protein sequences to ESM-2 embeddings.

    Args:
        smiles_list (list of str): A list of protein sequences.
        model: Pre-trained ESM-2 model.
        batch_converter: ESM-2 batch converter.

    Returns:
        np.ndarray: A NumPy array of ESM-2 embeddings for each protein sequence.
    """
    embeddings = []
    for smi in smiles_list:
        try:
            # Prepare data in the format expected by ESM-2
            data = [(str(i), smi) for i, smi in enumerate([smi])]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.cuda()
            
            # Generate embeddings
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33])
            embedding = results["representations"][33].cpu().numpy()
            
            # Use mean pooling over sequence length
            sequence_embedding = embedding.mean(axis=1)
            embeddings.append(sequence_embedding[0])
            
        except Exception as e:
            embeddings.append(np.zeros(1280))  # ESM-2 has 1280 dimensions
            
    return np.array(embeddings)


def get_balm_embeddings(targets_list, ligands_list, target_tokenizer, ligand_tokenizer, model, batch_size=128):
    """
    

     Compute BALM embeddings for protein targets and ligand sequences in batches.

    Args:
        targets_list (list of str): List of protein target sequences.
        ligands_list (list of str): List of protein ligand sequences.
        target_tokenizer: Tokenizer for the protein targets.
        ligand_tokenizer: Tokenizer for the protein ligands.
        model: BALM model used for generating embeddings.
        batch_size (int): The size of batches to process.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Protein embeddings, ligand embeddings, and cosine similarities.
    """
    # Prepare for batching
    protein_embeddings = []
    proteina_embeddings = []
    cosine_similarities = []
    
    # Ensure model is in evaluation mode
    model.eval()
    
    print("Computing embeddings...")
    # Create batches for more efficient processing
    for i in range(0, len(targets_list), batch_size):
        print(f"Processing batch {i // batch_size + 1}/{len(targets_list) // batch_size + 1}...")
        # Get the batch
        batch_targets = [" ".join(seq) for seq in targets_list[i:i + batch_size]]
        batch_ligands = [" ".join(seq) for seq in ligands_list[i:i + batch_size]]
        
        try:

            # Tokenize the batch
            target_inputs = target_tokenizer(batch_targets, return_tensors="pt", padding=True, truncation=True).to("cuda")
            ligand_inputs = ligand_tokenizer(batch_ligands, return_tensors="pt", padding=True, truncation=True).to("cuda")
        
            # Prepare inputs for the model
            inputs = {
                "protein_input_ids": target_inputs["input_ids"],
                "protein_attention_mask": target_inputs["attention_mask"],
                "proteina_input_ids": ligand_inputs["input_ids"],
                "proteina_attention_mask": ligand_inputs["attention_mask"],
            }
        
            # Run the model in batches
            with torch.no_grad():  # Disable gradient calculations for efficiency
                predictions = model(inputs)
        
            # Collect results
            protein_embeddings += [embedding.squeeze().detach().cpu().numpy() for embedding in predictions["protein_embedding"]]
            proteina_embeddings += [embedding.squeeze().detach().cpu().numpy() for embedding in predictions["proteina_embedding"]]
            cosine_similarities += [cos_sim.squeeze().detach().cpu().numpy() for cos_sim in predictions["cosine_similarity"]]
        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {str(e)}")
            # Add zero embeddings for failed batches
            embedding_dim = model.config.hidden_size  # typically 768 for BERT-based models
            batch_size_actual = len(batch_targets)
            protein_embeddings += [np.zeros(embedding_dim) for _ in range(batch_size_actual)]
            proteina_embeddings += [np.zeros(embedding_dim) for _ in range(batch_size_actual)]
            cosine_similarities += [0.0 for _ in range(batch_size_actual)]
    protein_embeddings = np.array(protein_embeddings)
    proteina_embeddings = np.array(proteina_embeddings)
    cosine_similarities = np.array(cosine_similarities)
    
    return protein_embeddings, proteina_embeddings, cosine_similarities


def split_data(X, y, smiles, test_size=0.8, seed=42):
    """
    Split data into training and testing sets, along with SMILES strings.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target values.
        smiles (list): Protein sequences.
        test_size (float): Proportion of the data to include in the test set.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple: Training and testing feature matrices, target values, and SMILES strings.
    """
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
        X, y, smiles, test_size=test_size, random_state=seed
    )
    return (torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
            smiles_train, smiles_test)

# GP and 
class ExactGPModelECFP8(gpytorch.models.ExactGP):
    """
    Gaussian Process model for ESM-2 embeddings using the RBF kernel.

    Args:
        train_x (torch.Tensor): Training feature data.
        train_y (torch.Tensor): Training target values.
        likelihood: Gaussian likelihood for the GP model.
    """
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModelECFP8, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # Changed from TanimotoKernel to RBFKernel since we're using protein embeddings
        self.covar_module = gpytorch.kernels.ScaleKernel(RBFKernel())
    def forward(self, x):
        """
        Forward pass of the GP model.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            gpytorch.distributions.MultivariateNormal: Predicted distribution over outputs.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPModelChemBERTa2(gpytorch.models.ExactGP):
    """
    Gaussian Process model for ESM-2 using the RBF kernel.

    Args:
        train_x (torch.Tensor): Training feature data.
        train_y (torch.Tensor): Training target values.
        likelihood: Gaussian likelihood for the GP model.
    """
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModelChemBERTa2, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(RBFKernel())

    def forward(self, x):
        """
        Forward pass of the GP model.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            gpytorch.distributions.MultivariateNormal: Predicted distribution over outputs.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPModelBALM(gpytorch.models.ExactGP):
    """
    Gaussian Process model for BALM embeddings using the RBF kernel.

    Args:
        train_x (torch.Tensor): Training feature data.
        train_y (torch.Tensor): Training target values.
        likelihood: Gaussian likelihood for the GP model.
    """
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModelBALM, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(RBFKernel())

    def forward(self, x):
        """
        Forward pass of the GP model.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            gpytorch.distributions.MultivariateNormal: Predicted distribution over outputs.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gp_model(
    X_train, y_train, X_test, y_test, smiles_test, model_class, learning_rate=0.1, epochs=50
):
    """
    Train a Gaussian Process model and log training progress to Weights & Biases.

    Args:
        X_train (torch.Tensor): Training feature matrix.
        y_train (torch.Tensor): Training target values.
        X_test (torch.Tensor): Test feature matrix.
        y_test (torch.Tensor): Test target values.
        smiles_test (list): List of protein strings for the test set.
        model_class (class): The class of the GP model to train.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.

    Returns:
        Tuple: RMSE, R-squared, Spearman correlation, and Pearson correlation.
    """
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = model_class(X_train, y_train, likelihood)

    # Training mode
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = ExactMarginalLogLikelihood(likelihood, model)

    # Initialize training logging for W&B
    for i in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()

        # Log the loss to W&B
        wandb.log({"epoch": i+1, "train/loss": loss.item()})

    # Evaluation mode
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        preds = likelihood(model(X_test)).mean

    preds = preds.detach().numpy()
    y_test = y_test.detach().numpy()

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    spearman_corr, _ = spearmanr(y_test, preds)
    pearson_corr, _ = pearsonr(y_test, preds)

    # Log evaluation metrics to W&B
    wandb.log({"test/rmse": rmse, "test/r2": r2, "test/spearman": spearman_corr, "test/pearson": pearson_corr})

    # Save predictions and actual values to a CSV
    predictions_df = pd.DataFrame({
        "protein": [None] * len(y_test),
        "proteina": smiles_test,
        "label": y_test,
        "prediction": preds
    })
    predictions_filename = "test_prediction.csv"
    predictions_df.to_csv(predictions_filename, index=False)

    # Create W&B artifact for the predictions
    artifact = wandb.Artifact("test_prediction", type="prediction")
    artifact.add_file(predictions_filename)
    wandb.log_artifact(artifact)

    return rmse, r2, spearman_corr, pearson_corr


def main():
    """
    Main function to load the dataset, train the model, and log results.
    """
    args = argument_parser()

    # Load dataset
    data = load_dataset("BALM/BALM-benchmark", args.dataset, split="train").to_pandas()

    # Initialize Weights & Biases logging
    wandb_entity = os.getenv("WANDB_ENTITY", "")
    wandb_project = os.getenv("WANDB_PROJECT_NAME", "")

    wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        name=f"gp__emb_{args.embedding_type}__test_{args.test_size}__lr_{args.lr}__data_{args.dataset}__seed_{args.random_seed}",
        config=args,
    )

    # Extract SMILES and target (Y)
    targets = data["Target"].to_list()
    smiles = data["proteina"].to_list()
    y = data["Y"].to_numpy()

    if args.debug:
        targets = targets[:100]
        smiles = smiles[:100]
        y = y[:100]
    # Load ESM-2 model (do this once, outside the conditions)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().cuda()
    batch_converter = alphabet.get_batch_converter()

    if args.embedding_type == "ESM2":
        # Convert SMILES to ECFP8 fingerprints
        X = smiles_to_ecfp8_fingerprint(smiles)
        

        # Split data
        X_train, X_test, y_train, y_test, smiles_train, smiles_test = split_data(
            X, y, smiles, test_size=args.test_size, seed=args.random_seed
        )

        model_class = ExactGPModelECFP8
        # Train and evaluate ECFP8-based GP
        rmse, r2, spearman, pearson = train_gp_model(
            X_train,
            y_train,
            X_test,
            y_test,
            smiles_test,
            ExactGPModelECFP8,
            args.lr,
            args.epochs,
        )
        
    elif args.embedding_type == "ESM2-proteina":
        # Convert second protein sequences to ESM2 embeddings
        X = smiles_to_chemberta_embedding(smiles, model, batch_converter)  # Now using ESM2 embeddings for second protein

        X_train, X_test, y_train, y_test, smiles_train, smiles_test = split_data(
            X, y, smiles, test_size=args.test_size, seed=args.random_seed
        )

        model_class = ExactGPModelChemBERTa2  # Now using ESM model for second protein
        
        
    elif args.embedding_type.startswith("BALM"):
        config_filepath = "default_configs/balm_peft.yaml"
        configs = Configs(**common_utils.load_yaml(config_filepath))

        # Load the model
        model = BALM(configs.model_configs)
        model = load_trained_model(model, configs.model_configs, is_training=False)
        model.to("cuda")

        # Use ESM-2 embeddings for both proteins
        target_embeddings = smiles_to_ecfp8_fingerprint(targets)  # ESM2 for first protein
        ligand_embeddings = smiles_to_chemberta_embedding(smiles, model, batch_converter)  # ESM2 for second protein
        
        # Calculate cosine similarities between ESM2 embeddings
        cosine_similarities = torch.nn.functional.cosine_similarity(
            torch.tensor(target_embeddings), 
            torch.tensor(ligand_embeddings)
        ).numpy()

        
        if args.embedding_type == "BALM-ligand":
            X = ligand_embeddings
        if args.embedding_type == "BALM-concat":
            X = np.concatenate((target_embeddings, ligand_embeddings), axis=1)
        if args.embedding_type == "BALM-sum":
            X = target_embeddings + ligand_embeddings
        if args.embedding_type == "BALM-subtract":
            X = target_embeddings - ligand_embeddings
        if args.embedding_type == "BALM-cosine":
            X = cosine_similarities

        X_train, X_test, y_train, y_test, smiles_train, smiles_test = split_data(
            X, y, smiles, test_size=args.test_size, seed=args.random_seed
        )

        model_class = ExactGPModelBALM

    # Train and evaluate ChemBERTa2-based GP
    rmse, r2, spearman, pearson = train_gp_model(
        X_train,
        y_train,
        X_test,
        y_test,
        smiles_test,
        model_class,
        args.lr,
        args.epochs,
    )

    # Print results
    print(
        f"{args.embedding_type} GP Model - RMSE: {rmse}, "
        f"R^2: {r2}, "
        f"Spearman: {spearman}, "
        f"Pearson: {pearson}"
    )

    # Finish W&B run
    wandb.finish()


if __name__ == "__main__":
    main()
