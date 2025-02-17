import torch
from torch import nn
from torch.nn import functional as F
import esm

from balm.configs import ModelConfigs
from balm.models.base_model import BaseModel

class BaselineModel(BaseModel):
    """
    BaselineModel model extends BaseModel to concatenate protein encodings from ESM-2.
    This model takes the embeddings from both proteins, concatenates them, and processes them further.
    When fine-tuning is enabled, the embeddings are trainable through PEFT methods.

    Attributes:
        model_configs (ModelConfigs): The configuration object for the model.
        protein_model: The ESM-2 model for first protein (potentially fine-tuned).
        proteina_model: The ESM-2 model for second protein (potentially fine-tuned).
        protein_embedding_size (int): The size of the first protein embeddings (1280 for ESM-2).
        proteina_embedding_size (int): The size of the second protein embeddings (1280 for ESM-2).
    """

    def __init__(
        self,
        model_configs: ModelConfigs,
        protein_embedding_size=1280,
        proteina_embedding_size=1280,
    ):
        super(BaselineModel, self).__init__(
            model_configs, protein_embedding_size, proteina_embedding_size
        )

        # Projection layers
        self.linear_projection = nn.Linear(
            self.protein_embedding_size + self.proteina_embedding_size,
            model_configs.model_hyperparameters.projected_size,
        )
        self.dropout = nn.Dropout(model_configs.model_hyperparameters.projected_dropout)
        self.out = nn.Linear(model_configs.model_hyperparameters.projected_size, 1)

        self.print_trainable_params()
        self.loss_fn = nn.MSELoss()

    def forward(self, batch_input, **kwargs):
        forward_output = {}

        # Process sequences
        protein_data = [(str(i), seq) for i, seq in enumerate(batch_input["protein_sequences"])]
        proteina_data = [(str(i), seq) for i, seq in enumerate(batch_input["proteina_sequences"])]
        
        _, _, protein_tokens = self.batch_converter(protein_data)
        _, _, proteina_tokens = self.batch_converter(proteina_data)
        
        protein_tokens = protein_tokens.to(self.device)
        proteina_tokens = proteina_tokens.to(self.device)

        # Get embeddings - conditional on fine-tuning mode
        if self.fine_tuning_method is None:
            with torch.no_grad():
                protein_results = self.protein_model(protein_tokens, repr_layers=[33])
                proteina_results = self.proteina_model(proteina_tokens, repr_layers=[33])
        else:
            protein_results = self.protein_model(protein_tokens, repr_layers=[33])
            proteina_results = self.proteina_model(proteina_tokens, repr_layers=[33])

        # Mean pool over sequence length
        protein_embedding = protein_results["representations"][33].mean(dim=1)
        proteina_embedding = proteina_results["representations"][33].mean(dim=1)

        # Concatenate embeddings
        concatenated_embedding = torch.cat((protein_embedding, proteina_embedding), 1)

        # Process through dense layers
        projected_embedding = F.relu(self.linear_projection(concatenated_embedding))
        projected_embedding = self.dropout(projected_embedding)
        logits = self.out(projected_embedding)

        if "labels" in batch_input and batch_input["labels"] is not None:
            forward_output["loss"] = self.loss_fn(logits, batch_input["labels"])

        forward_output["protein_embedding"] = protein_embedding
        forward_output["proteina_embedding"] = proteina_embedding
        forward_output["logits"] = logits.squeeze(-1)

        return forward_output