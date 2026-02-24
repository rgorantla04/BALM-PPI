from typing import Tuple, Optional
import os
import torch
from huggingface_hub import hf_hub_download
import esm
from peft import get_peft_model

from balm.configs import ModelConfigs
from balm.models.base_model import BaseModel

def load_trained_model(model: BaseModel, model_configs: ModelConfigs, is_training: bool) -> BaseModel:
    """
    Load and configure ESM-2 models with optional fine-tuning support.

    Args:
        model (BaseModel): The model instance to load the checkpoint into.
        model_configs (ModelConfigs): Configuration object containing model-related settings.
        is_training (bool): Flag indicating whether the model is being loaded for training or evaluation.

    Returns:
        BaseModel: The configured model ready for training or evaluation.
    """
    print(f"Loading ESM-2 models for both proteins")
    
    # Load ESM-2 models
    protein_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    proteina_model, _ = esm.pretrained.esm2_t33_650M_UR50D()
    
    # Move models to appropriate device
    protein_model = protein_model.to(model.device)
    proteina_model = proteina_model.to(model.device)
    
    # Set up batch converter
    batch_converter = alphabet.get_batch_converter()
    
    # Configure models based on fine-tuning settings
    if hasattr(model_configs, 'fine_tuning_method') and model_configs.fine_tuning_method:
        try:
            # Apply PEFT configuration from BaseModel
            model.protein_model = model._setup_peft_model(protein_model)
            model.proteina_model = model._setup_peft_model(proteina_model)
        except Exception as e:
            raise ValueError(f"Error applying fine-tuning configuration: {str(e)}")
    else:
        # Traditional frozen approach
        model.protein_model = protein_model
        model.proteina_model = proteina_model
        if is_training:
            # Only train projection layers
            for name, params in model.named_parameters():
                if "projection" not in name:
                    params.requires_grad = False
    
    model.batch_converter = batch_converter
    
    # Load checkpoint if provided
    if model_configs.checkpoint_path:
        load_checkpoint_state(model, model_configs.checkpoint_path)
    
    # Set evaluation mode if not training
    if not is_training:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    
    # Print parameter info
    model.print_trainable_params()
    
    return model

def load_checkpoint_state(model: BaseModel, checkpoint_path: str):
    """
    Load model checkpoint, handling both standard and PEFT states.

    Args:
        model (BaseModel): Model instance to load state into.
        checkpoint_path (str): Path to the checkpoint directory or file.
    """
    if os.path.isdir(checkpoint_path):
        # Load PEFT state if available
        if hasattr(model.protein_model, 'load_pretrained'):
            model.protein_model.load_pretrained(checkpoint_path + "_protein")
            model.proteina_model.load_pretrained(checkpoint_path + "_proteina")
        
        # Load projection layers state
        state_dict = torch.load(f"{checkpoint_path}_state.pt")
        model.load_state_dict(state_dict['projection_state'], strict=False)
    else:
        # Load complete state dict
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict, strict=False)

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