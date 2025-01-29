import torch
from torch import nn
from torch.nn import functional as F
import esm
from balm.configs import ModelConfigs


class BaseModel(nn.Module):
    def __init__(self, model_configs, protein_embedding_size=1280, proteina_embedding_size=1280):
        super(BaseModel, self).__init__()
        self.model_configs = model_configs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load ESM-2 models
        self.protein_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.proteina_model, _ = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Freeze ESM models
        for param in self.protein_model.parameters():
            param.requires_grad = False
        for param in self.proteina_model.parameters():
            param.requires_grad = False

        self.protein_embedding_size = protein_embedding_size
        self.proteina_embedding_size = proteina_embedding_size
    def print_trainable_params(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for name, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                print(name)
                trainable_params += num_params

        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )

    def forward(self, batch_input):
        """
        Base forward pass implementation.
        Should be overridden by child classes with specific implementation.
        """
        raise NotImplementedError("Forward method must be implemented by child classes")