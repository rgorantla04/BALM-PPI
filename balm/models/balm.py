import torch
from torch import nn
from torch.nn import functional as F
import esm

from balm.configs import ModelConfigs
from balm.models.base_model import BaseModel

class BALM(BaseModel):
    def __init__(
        self,
        model_configs: ModelConfigs,
        protein_embedding_size=1280,
        proteina_embedding_size=1280,
    ):
        super(BALM, self).__init__(
            model_configs, protein_embedding_size, proteina_embedding_size
        )

        # Modified projection layers with residual connections
        self.protein_projection = nn.Sequential(
            nn.Linear(1280, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
        
        self.proteina_projection = nn.Sequential(
            nn.Linear(1280, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )

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

        protein_embedding = protein_results["representations"][33].mean(dim=1)
        proteina_embedding = proteina_results["representations"][33].mean(dim=1)

        # Project embeddings
        protein_projected = self.protein_projection(protein_embedding)
        proteina_projected = self.proteina_projection(proteina_embedding)

        # Apply L2 normalization
        protein_projected = F.normalize(protein_projected, p=2, dim=-1)
        proteina_projected = F.normalize(proteina_projected, p=2, dim=-1)

        # Compute cosine similarity
        cosine_similarity = F.cosine_similarity(protein_projected, proteina_projected)

        if "labels" in batch_input and batch_input["labels"] is not None:
            forward_output["loss"] = self.loss_fn(cosine_similarity, batch_input["labels"])

        forward_output["cosine_similarity"] = cosine_similarity
        forward_output["protein_embedding"] = protein_embedding
        forward_output["proteina_embedding"] = proteina_embedding

        return forward_output

    @staticmethod
    def cosine_similarity_to_pkd(cosine_similarity, pkd_upper_bound, pkd_lower_bound):
        """Convert cosine similarity to pKd values with robust numerical handling"""
        # Ensure tensor is on CPU and handle nan/inf
        cosine_similarity = torch.nan_to_num(cosine_similarity, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Clamp values to valid cosine similarity range
        cosine_similarity = torch.clamp(cosine_similarity, -1.0, 1.0)
        
        # Convert to pKd range
        pkd_range = pkd_upper_bound - pkd_lower_bound
        scaled_value = ((cosine_similarity + 1.0) / 2.0) * pkd_range + pkd_lower_bound

        
        # Final clipping to ensure values stay in valid range
        return torch.clamp(scaled_value, pkd_lower_bound, pkd_upper_bound)