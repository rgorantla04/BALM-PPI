import torch
from torch import nn
from torch.nn import functional as F
import esm
from peft import LoraConfig, LoHaConfig, LoKrConfig, IA3Config, get_peft_model, TaskType
from balm.configs import ModelConfigs

PEFT_CONFIGS = {
    "lora": LoraConfig,
    "loha": LoHaConfig,
    "lokr": LoKrConfig,
    "ia3": IA3Config
}

class BaseModel(nn.Module):
    def __init__(self, model_configs, protein_embedding_size=1280, proteina_embedding_size=1280):
        super(BaseModel, self).__init__()
        self.model_configs = model_configs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Extract fine-tuning configuration
        self.fine_tuning_method = getattr(model_configs, 'fine_tuning_method', None)
        self.fine_tuning_config = getattr(model_configs, 'fine_tuning_config', {})
        
        # Load and setup models
        self.protein_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.proteina_model, _ = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Initialize models with or without fine-tuning
        self.protein_model = self._initialize_model(self.protein_model)
        self.proteina_model = self._initialize_model(self.proteina_model)

        self.protein_embedding_size = protein_embedding_size
        self.proteina_embedding_size = proteina_embedding_size

    def _initialize_model(self, model):
        """Initialize model with or without fine-tuning"""
        if self.fine_tuning_method is None:
            # Traditional frozen approach
            for param in model.parameters():
                param.requires_grad = False
            return model
        else:
            # Apply PEFT
            peft_config = self._setup_peft_config(
                self.fine_tuning_method,
                self.fine_tuning_config
            )
            return get_peft_model(model, peft_config)

    def _setup_peft_config(self, method, config):
        """Setup PEFT configuration"""
        if method not in PEFT_CONFIGS:
            raise ValueError(f"Unsupported fine-tuning method: {method}")
            
        base_config = {
            "task_type": TaskType.TOKEN_CLS,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"],
            "inference_mode": False,
            "r": config.get("r", 8),
            "alpha": config.get("alpha", 16),
            "dropout": config.get("dropout", 0.1)
        }
        base_config.update(config)
        return PEFT_CONFIGS[method](**base_config)

    def set_fine_tuning_mode(self, training=True):
        """Set models to training or inference mode"""
        if hasattr(self.protein_model, 'set_inference_mode'):
            self.protein_model.set_inference_mode(not training)
            self.proteina_model.set_inference_mode(not training)

    def print_trainable_params(self):
        """Print trainable parameters with PEFT support"""
        trainable_params = 0
        all_param = 0
        
        for name, param in self.named_parameters():
            num_params = param.numel()
            
            # Handle DS Zero 3
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Handle 4-bit params
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                print(name)
                trainable_params += num_params

        # Print summary
        print(
            f"trainable params: {trainable_params:,d} || "
            f"all params: {all_param:,d} || "
            f"trainable%: {100 * trainable_params / all_param:.2f}"
        )
        
        # Print PEFT-specific info if applicable
        if self.fine_tuning_method is not None:
            if hasattr(self.protein_model, "print_trainable_parameters"):
                print("\nProtein Model PEFT Parameters:")
                self.protein_model.print_trainable_parameters()
                print("\nProteinA Model PEFT Parameters:")
                self.proteina_model.print_trainable_parameters()

    def forward(self, batch_input):
        """Base forward pass implementation"""
        raise NotImplementedError("Forward method must be implemented by child classes")