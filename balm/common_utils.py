import os
import random
from datetime import datetime
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import yaml
from torch import nn
from transformers import set_seed

from .configs import Configs


def setup_experiment_folder(outputs_dir: str) -> Tuple[str, str, str]:
    """
    Setup experiment directories including PEFT configurations.
    
    Returns:
        Tuple[str, str, str]: outputs_dir, checkpoint_dir, peft_config_dir
    """
    now = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    outputs_dir = os.path.join(outputs_dir, now)
    checkpoint_dir = os.path.join(outputs_dir, "checkpoint")
    peft_config_dir = os.path.join(checkpoint_dir, "peft_config")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(peft_config_dir, exist_ok=True)
    
    return outputs_dir, checkpoint_dir, peft_config_dir

def count_parameters_by_type(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters by type (base, PEFT, other trainable).
    """
    stats = {
        "base_params": 0,
        "peft_params": 0,
        "other_trainable": 0,
        "total_trainable": 0
    }
    
    for name, param in model.named_parameters():
        if hasattr(param, "is_peft_param"):
            stats["peft_params"] += param.numel()
        elif "projection" in name:
            stats["other_trainable"] += param.numel()
        else:
            stats["base_params"] += param.numel()
            
        if param.requires_grad:
            stats["total_trainable"] += param.numel()
            
    return stats

def setup_training_environment(
    configs: Configs,
    device: Optional[str] = None,
    seed: int = 42
) -> Tuple[torch.device, bool]:
    """
    Comprehensive training environment setup.
    """
    # Setup device
    device, use_half = setup_esm_device(device)
    
    # Setup random seed
    setup_random_seed(seed)
    
    # Enable gradient checkpointing if fine-tuning
    if hasattr(configs.model_configs, 'fine_tuning_method') and configs.model_configs.fine_tuning_method:
        torch.backends.cudnn.benchmark = False
        if use_half:
            print("Using half precision with fine-tuning")
    
    return device, use_half


def setup_esm_device(device: Optional[str] = None) -> Tuple[torch.device, bool]:
    """
    Utility function to setup the device for ESM-2 model.
    
    Args:
        device (Optional[str]): Device specification.
        
    Returns:
        Tuple[torch.device, bool]: Device and whether to use half precision
    """
    if torch.cuda.is_available():
        device = f"cuda:{device}"
        use_half = True  # ESM-2 can use half precision on GPU
    else:
        try:
            if torch.backends.mps.is_available():
                device = "mps"
                use_half = False  # Don't use half precision on MPS
        except:
            device = "cpu"
            use_half = False  # Don't use half precision on CPU
    return torch.device(device), use_half


def setup_random_seed(seed: int, is_deterministic: bool = True) -> None:
    """
    Utility function to setup random seed. Apply this function early on the training script.

    Args:
        seed (int): Integer indicating the desired seed.
        is_deterministic (bool, optional): Set deterministic flag of CUDNN. Defaults to True.
    """
    # set the seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if is_deterministic is True:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    set_seed(seed)


def count_parameters(model: nn.Module) -> int:
    """
    Utility function to calculate the number of parameters in a model.

    Args:
        model (nn.Module): Model in question.

    Returns:
        int: Number of parameters of the model in question.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_yaml(filepath: str) -> dict:
    """
    Utility function to load yaml file, mainly for config files.

    Args:
        filepath (str): Path to the config file.

    Raises:
        exc: Stop process if there is a problem when loading the file.

    Returns:
        dict: Training configs.
    """
    with open(filepath, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise exc

def save_training_configs(configs: Configs, output_dir: str):
    """
    Save training configs including PEFT settings.
    """
    # Save main config
    config_dict = configs.dict()
    
    # Handle PEFT config separately
    if hasattr(configs.model_configs, 'fine_tuning_method') and configs.model_configs.fine_tuning_method:
        peft_config = get_peft_config_to_save(configs.model_configs.fine_tuning_config)
        config_dict['peft_config'] = peft_config
    
    filepath = os.path.join(output_dir, "configs.yaml")
    with open(filepath, "w") as file:
        yaml.dump(config_dict, file)


def delete_files_in_directory(directory: str):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path} due to {e}")

def rescale_pkd(value, original_min, original_max, target_min, target_max):
    """
    Rescale pKd values from one range to another.
    
    Args:
        value: Value to rescale
        original_min: Original minimum value
        original_max: Original maximum value
        target_min: Target minimum value
        target_max: Target maximum value
    
    Returns:
        Rescaled value
    """
    return ((value - original_min) / (original_max - original_min)) * (target_max - target_min) + target_min