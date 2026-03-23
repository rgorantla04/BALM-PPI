"""
Reproducibility utilities for BALM-PPI experiments.
Ensures consistent random seeds across different libraries.
"""

import os
import random
import numpy as np
import torch


def setup_reproducibility(seed: int = 42) -> None:
    """
    Setup complete reproducibility with consistent random seeds.
    
    Args:
        seed: Random seed value (default: 42)
    """
    print(f"[LOCK] Setting up reproducibility with seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    print("✅ Reproducibility setup complete")
    print(f"✅ Reproducibility setup complete")
