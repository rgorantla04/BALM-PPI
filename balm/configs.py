from enum import Enum
from typing import List, Optional, Union, Dict
from pydantic import BaseModel, validator, model_validator

class TargetModules(str, Enum):
    """ESM-2 target modules for PEFT methods"""
    ATTENTION = "attention"  # q_proj, k_proj, v_proj, o_proj
    FFN = "ffn"  # fc1, fc2
    ALL = "all"  # both attention and ffn

class FineTuningType(str, Enum):
    """Extended fine-tuning types including PEFT methods"""
    baseline = "baseline"
    projection = "projection"
    lora = "lora"
    loha = "loha"
    lokr = "lokr"
    ia3 = "ia3"

class PeftConfig(BaseModel):
    """Base configuration for PEFT methods"""
    method: FineTuningType
    target_modules: Union[TargetModules, List[str]] = TargetModules.ATTENTION
    rank: Optional[int] = 8  # for LoRA, LoHa, LoKr
    alpha: Optional[float] = 16  # for LoRA, LoHa, LoKr
    dropout: float = 0.1
    bias: str = "none"
    
    @validator('rank')
    def validate_rank(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Rank must be positive")
        return v
    
    @validator('alpha')
    def validate_alpha(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Alpha must be positive")
        return v

class ModelHyperparameters(BaseModel):
    learning_rate: float = 0.001
    protein_max_seq_len: int = 2048
    proteina_max_seq_len: int = 2048
    warmup_steps_ratio: float = 0.06
    gradient_accumulation_steps: int = 32
    projected_size: int = 256
    projected_dropout: float = 0.5
    relu_before_cosine: bool = False
    init_noise_sigma: float = 1
    sigma_lr: float = 0.01
    esm_layer: int = 33
    mean_pool: bool = True

class ModelConfigs(BaseModel):
    esm_model_name: str = "esm2_t33_650M_UR50D"
    protein_model_name_or_path: Optional[str] = None
    proteina_model_name_or_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    model_hyperparameters: ModelHyperparameters
    protein_fine_tuning_type: Optional[FineTuningType] = None
    proteina_fine_tuning_type: Optional[FineTuningType] = None
    protein_peft_config: Optional[PeftConfig] = None
    proteina_peft_config: Optional[PeftConfig] = None
    loss_function: str
    
    
    @model_validator(mode='after')
    def validate_peft_configs(self) -> 'ModelConfigs':
        p_type = self.protein_fine_tuning_type
        pa_type = self.proteina_fine_tuning_type
        p_config = self.protein_peft_config
        pa_config = self.proteina_peft_config
        
        if p_type in [FineTuningType.lora, FineTuningType.loha, 
                     FineTuningType.lokr, FineTuningType.ia3]:
            if not p_config:
                raise ValueError(f"PEFT config required for {p_type}")
                
        if pa_type in [FineTuningType.lora, FineTuningType.loha, 
                      FineTuningType.lokr, FineTuningType.ia3]:
            if not pa_config:
                raise ValueError(f"PEFT config required for {pa_type}")
        
        return self

# DatasetConfigs and TrainingConfigs remain unchanged
class DatasetConfigs(BaseModel):
    dataset_name: str
    harmonize_affinities_mode: Optional[str]
    split_method: str = "random"
    train_ratio: Optional[float] = None

class TrainingConfigs(BaseModel):
    random_seed: int = 1234
    device: int = 0
    epochs: int = 1
    batch_size: int = 4
    patience: int = 100
    min_delta: int = 0.005
    outputs_dir: str = "outputs"

class Configs(BaseModel):
    model_configs: ModelConfigs
    dataset_configs: DatasetConfigs
    training_configs: TrainingConfigs