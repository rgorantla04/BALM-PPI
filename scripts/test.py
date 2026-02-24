import argparse
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv("env/.env")

import time
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from balm import common_utils
from balm.configs import Configs
from balm.dataset import DataCollatorWithPadding
from balm.metrics import get_ci, get_pearson, get_rmse, get_spearman
from balm.model import BALM, BaselineModel
from balm.tokenization import pre_tokenize_unique_entities, tokenize_with_lookup


def argument_parser():
    """
    Parses command-line arguments for the BALM testing script.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="BALM")
    parser.add_argument("--config_filepath", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--test_data", type=str, required=True, help="Path to the test data CSV file.")
    parser.add_argument("--pkd_upper_bound", type=float, default=10.0, help="Upper bound for pKd scaling.")
    parser.add_argument("--pkd_lower_bound", type=float, default=1.999999995657055, help="Lower bound for pKd scaling.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint/", help="Directory containing model checkpoints.")
    args = parser.parse_args()
    return args


def get_checkpoint_name(configs: Configs):
    # Method abbreviations
    METHOD_ABBREV = {
        "lora": "lr",
        "loha": "lh",
        "lokr": "lk",
        "ia3": "ia"
    }
    
    hyperparams = []
    if configs.model_configs.peft_configs.enabled:
        method = configs.model_configs.peft_configs.protein.method
        hyperparams.append(f"peft_{METHOD_ABBREV.get(method, method)}")
        hyperparams.append(f"r{configs.model_configs.peft_configs.protein.rank}")
    
    hyperparams += [
        f"lr_{configs.model_configs.model_hyperparameters.learning_rate}",
        f"dim_{configs.model_configs.model_hyperparameters.projected_size}",
    ]
    return "_".join(hyperparams)


def load_model(configs, checkpoint_dir):
    # Initialize appropriate model
    if configs.model_configs.fine_tuning_method:
        model = BALM(configs.model_configs)
    else:
        model = BaselineModel(configs.model_configs)
    
    # Move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load checkpoint
    checkpoint_name = get_checkpoint_name(configs)
    checkpoint = torch.load(
        os.path.join(checkpoint_dir, checkpoint_name, "pytorch_model.bin"),
        map_location=device
    )
    model.load_state_dict(checkpoint)
    return model.eval()

def load_tokenizers(configs):
    """
    Loads the tokenizers for protein and proteina sequences.

    Args:
        configs (Configs): Configuration object containing model settings.

    Returns:
        tuple: A tuple containing the protein and proteina tokenizers.
    """
    protein_tokenizer = AutoTokenizer.from_pretrained(configs.model_configs.protein_model_name_or_path)
    proteina_tokenizer = AutoTokenizer.from_pretrained(configs.model_configs.proteina_model_name_or_path)

    return protein_tokenizer, proteina_tokenizer


def load_data(test_data, batch_size, protein_max_seq_len, proteina_max_seq_len):
    df = pd.read_csv(test_data)
    dataset = Dataset.from_pandas(df)
    
    data_collator = ESMDataCollator(
        batch_converter=model.batch_converter,
        protein_max_length=protein_max_seq_len,
        proteina_max_length=proteina_max_seq_len,
    )
    
    return DataLoader(
        dataset,
        shuffle=False,  # Changed to False for testing
        collate_fn=data_collator,
        batch_size=batch_size,
        pin_memory=True,
    )


def compute_metrics(labels, predictions, pkd_upper_bound, pkd_lower_bound):
    """
    Computes performance metrics including RMSE, Pearson, Spearman, and CI.

    Args:
        labels (Tensor): Ground truth labels.
        predictions (Tensor): Model predictions.
        pkd_upper_bound (float): Upper bound for pKd scaling.
        pkd_lower_bound (float): Lower bound for pKd scaling.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    pkd_range = pkd_upper_bound - pkd_lower_bound
    labels = (labels + 1) / 2 * pkd_range + pkd_lower_bound
    predictions = (predictions + 1) / 2 * pkd_range + pkd_lower_bound

    rmse = get_rmse(labels, predictions)
    pearson = get_pearson(labels, predictions)
    spearman = get_spearman(labels, predictions)
    ci = get_ci(labels, predictions)

    return {
        "rmse": rmse,
        "pearson": pearson,
        "spearman": spearman,
        "ci": ci,
    }


def main():
    """
    Main function to execute the testing process for BALM.

    It performs the following steps:
    1. Parses command-line arguments.
    2. Loads configuration settings and model checkpoints.
    3. Prepares the test dataset.
    4. Evaluates the model on the test data and computes performance metrics.
    """
    args = argument_parser()
    config_filepath = args.config_filepath
    configs = Configs(**common_utils.load_yaml(config_filepath))

    protein_max_seq_len = configs.model_configs.model_hyperparameters.protein_max_seq_len
    proteina_max_seq_len = configs.model_configs.model_hyperparameters.proteina_max_seq_len
    protein_tokenizer, proteina_tokenizer = load_tokenizers(configs)

    model = load_model(configs, args.checkpoint_dir)
    dataloader = load_data(
        args.test_data,
        configs.training_configs.batch_size,
        protein_tokenizer,
        proteina_tokenizer,
        protein_max_seq_len,
        proteina_max_seq_len,
    )

    pkd_upper_bound = args.pkd_upper_bound
    pkd_lower_bound = args.pkd_lower_bound

    start = time.time()
    all_labels = []
    all_predictions = []
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            batch = {key: value.to(model.protein_model.device) for key, value in batch.items()}
            outputs = model(batch)
            all_labels += [batch["labels"]]
            all_predictions += [outputs["cosine_similarity"]]
            if step % 10:
                print(
                    f"Time elapsed: {time.time()-start}s ; Processed: {step * configs.training_configs.batch_size}"
                )
    end = time.time()
    print(f"Finished! Time taken: {end - start}s")

    all_labels = torch.cat(all_labels, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    performance_metrics = compute_metrics(
        all_labels, all_predictions, pkd_upper_bound, pkd_lower_bound
    )
    print(performance_metrics)


if __name__ == "__main__":
    main()
