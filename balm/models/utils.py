from typing import Tuple
import os
import torch
from huggingface_hub import hf_hub_download
import esm  # Add ESM import

from balm.configs import ModelConfigs
from balm.models.base_model import BaseModel

def load_trained_model(model: BaseModel, model_configs: ModelConfigs, is_training: bool) -> BaseModel:
    """
    Load pre-trained ESM-2 models for both proteins and apply necessary adjustments.

    Args:
        model (BaseModel): The model instance to load the checkpoint into.
        model_configs (ModelConfigs): Configuration object containing model-related settings.
        is_training (bool): Flag indicating whether the model is being loaded for training or evaluation.

    Returns:
        BaseModel: The model loaded with ESM-2 models and prepared for either training or evaluation.
    """
    
    print(f"Loading ESM-2 models for both proteins")
    
    # Load ESM-2 model for both proteins (using the same architecture)
    protein_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    proteina_model, _ = esm.pretrained.esm2_t33_650M_UR50D()  # Using same model for second protein

    # Move models to appropriate device
    protein_model = protein_model.to(model.device)
    proteina_model = proteina_model.to(model.device)

    # Set up batch converter
    batch_converter = alphabet.get_batch_converter()

    # Assign models to the BALM model
    model.protein_model = protein_model
    model.proteina_model = proteina_model
    model.batch_converter = batch_converter

    # Configure the model for training or evaluation
    if is_training:
        # Freeze ESM-2 base parameters and only train projection layers
        for name, params in model.named_parameters():
            if "projection" not in name:
                params.requires_grad = False
        model.print_trainable_params()
    else:
        # Freeze all parameters for evaluation
        for name, params in model.named_parameters():
            params.requires_grad = False
        model.eval()

    return model

def load_pretrained_pkd_bounds(checkpoint_path: str) -> Tuple[float, float]:
    """
    Load pre-defined pKd bounds based on the specific checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint.

    Returns:
        Tuple[float, float]: The lower and upper bounds of pKd values for the given dataset.

    Raises:
        ValueError: If the checkpoint path does not match any known pKd scale.
    """
    if "bdb" in checkpoint_path:
        # BindingDB pKd scale
        pkd_lower_bound = 1.999999995657055
        pkd_upper_bound = 10.0
    elif "leakypdb" in checkpoint_path:
        pkd_lower_bound = 0.4
        pkd_upper_bound = 15.22
    elif "mpro" in checkpoint_path:
        pkd_lower_bound = 4.01
        pkd_upper_bound = 10.769216066691143
    else:
        # Raise an error if an unknown pKd scale is encountered
        raise ValueError(
            f"Unknown pKd scale, for {checkpoint_path}"
        )
    
    return pkd_lower_bound, pkd_upper_bound