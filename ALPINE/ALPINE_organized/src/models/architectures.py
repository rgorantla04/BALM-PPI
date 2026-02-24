"""
Model architectures for protein-protein binding affinity prediction.
Includes: Baseline, Model-1 (Frozen), and ALPINE (with LoRA).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from transformers import AutoModel, AutoTokenizer


class BALMProjectionHead(nn.Module):
    """
    Projection head for BALM architecture.
    Projects embeddings and computes cosine similarity.
    """
    
    def __init__(self, embedding_size: int, projected_size: int, projected_dropout: float):
        """
        Args:
            embedding_size: Input embedding dimension
            projected_size: Output projection dimension
            projected_dropout: Dropout rate
        """
        super().__init__()
        self.protein_projection = nn.Linear(embedding_size, projected_size)
        self.proteina_projection = nn.Linear(embedding_size, projected_size)
        self.dropout = nn.Dropout(projected_dropout)
        self.loss_fn = nn.MSELoss()
        print(f"✅ BALMProjectionHead initialized with embedding_size: {embedding_size}")
    
    def forward(self, protein_embedding: torch.Tensor, proteina_embedding: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Dict:
        """
        Args:
            protein_embedding: Protein 1 embeddings
            proteina_embedding: Protein 2 embeddings
            labels: Optional target labels for loss computation
            
        Returns:
            Dictionary with cosine_similarity and optional loss
        """
        protein_embedding = self.dropout(protein_embedding)
        proteina_embedding = self.dropout(proteina_embedding)
        
        protein_projected = self.protein_projection(protein_embedding)
        proteina_projected = self.proteina_projection(proteina_embedding)
        
        # L2 normalization
        protein_projected = F.normalize(protein_projected, p=2, dim=1)
        proteina_projected = F.normalize(proteina_projected, p=2, dim=1)
        
        # Cosine similarity
        cosine_similarity = F.cosine_similarity(protein_projected, proteina_projected)
        cosine_similarity = torch.clamp(cosine_similarity, -0.9999, 0.9999)
        
        output = {"cosine_similarity": cosine_similarity}
        
        if labels is not None:
            loss = self.loss_fn(cosine_similarity, labels)
            output["loss"] = loss
        
        return output


class BALMForRegression(nn.Module):
    """
    BALM regression model for frozen ESM-2 backbone with trainable projection head.
    Uses pre-computed embeddings.
    """
    
    def __init__(self, embedding_size: int, projected_size: int, 
                 projected_dropout: float, pkd_bounds: Tuple[float, float]):
        """
        Args:
            embedding_size: Embedding dimension from PLM
            projected_size: Projection head output dimension
            projected_dropout: Dropout rate
            pkd_bounds: Tuple of (min_pkd, max_pkd) for scaling
        """
        super().__init__()
        self.projection_head = BALMProjectionHead(embedding_size, projected_size, projected_dropout)
        self.pkd_lower, self.pkd_upper = pkd_bounds
        self.pkd_range = self.pkd_upper - self.pkd_lower
        print(f"✅ BALMForRegression model initialized.")
    
    def forward(self, batch_input: Dict) -> Dict:
        """
        Args:
            batch_input: Dictionary with protein embeddings and metadata
            
        Returns:
            Dictionary with predictions and metadata
        """
        protein_emb = batch_input["protein_embedding"]
        proteina_emb = batch_input["proteina_embedding"]
        labels = batch_input.get("labels")
        
        proj_output = self.projection_head(protein_emb, proteina_emb, labels)
        
        # Return all outputs and metadata
        output = {
            "cosine_similarity": proj_output["cosine_similarity"],
            "original_pkds": batch_input["original_pkds"],
            "pdb_groups": batch_input["pdb_groups"],
            "subgroups": batch_input["subgroups"],
            "source_dataset": batch_input["source_dataset"]
        }
        
        if "loss" in proj_output:
            output["loss"] = proj_output["loss"]
        
        return output


class FastBaselinePPIModel(nn.Module):
    """
    Fast baseline model using concatenated embeddings.
    """
    
    def __init__(self, embedding_size: int = 1280, projected_size: int = 256, 
                 projected_dropout: float = 0.1, device: str = "auto"):
        """
        Args:
            embedding_size: Input embedding dimension
            projected_size: Hidden projection dimension
            projected_dropout: Dropout rate
            device: Device to use
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
            if device == "auto" else torch.device(device)
        
        self.linear_projection = nn.Linear(2 * embedding_size, projected_size)
        self.dropout = nn.Dropout(projected_dropout)
        self.out = nn.Linear(projected_size, 1)
        self.loss_fn = nn.MSELoss()
        self.to(self.device)
    
    def forward(self, protein1_embeddings: torch.Tensor, protein2_embeddings: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Dict:
        """
        Args:
            protein1_embeddings: First protein embeddings
            protein2_embeddings: Second protein embeddings
            labels: Optional target labels
            
        Returns:
            Dictionary with logits and optional loss
        """
        concat_emb = torch.cat(
            (protein1_embeddings.to(self.device), protein2_embeddings.to(self.device)), 
            dim=1
        )
        proj_emb = F.relu(self.dropout(self.linear_projection(concat_emb)))
        logits = self.out(proj_emb).squeeze(-1)
        
        output = {"logits": logits}
        
        if labels is not None:
            output["loss"] = self.loss_fn(logits, labels.to(self.device))
        
        return output


class BALMForLoRAFinetuning(nn.Module):
    """
    BALM model with LoRA fine-tuning on ESM-2.
    Used for ALPINE experiments.
    """
    
    def __init__(self, esm_model: nn.Module, esm_tokenizer, projected_size: int, 
                 projected_dropout: float, pkd_bounds: Tuple[float, float]):
        """
        Args:
            esm_model: Pre-configured PEFT model with LoRA
            esm_tokenizer: Tokenizer for sequences
            projected_size: Projection head output dimension
            projected_dropout: Dropout rate
            pkd_bounds: Tuple of (min_pkd, max_pkd) for scaling
        """
        super().__init__()
        self.esm_model = esm_model
        self.esm_tokenizer = esm_tokenizer
        self.embedding_size = self.esm_model.config.hidden_size
        self.projection_head = BALMProjectionHead(self.embedding_size, projected_size, projected_dropout)
        self.pkd_lower, self.pkd_upper = pkd_bounds
        self.pkd_range = self.pkd_upper - self.pkd_lower
        self.cls_token = self.esm_tokenizer.cls_token
        
        print(f"✅ BALM LoRA model initialized using CLS token: {self.cls_token}")
    
    def _get_esm_embeddings(self, sequences: List[str]) -> torch.Tensor:
        """
        Get embeddings from ESM-2 with LoRA.
        Uses Seq1 + CLS + CLS + Seq2 and mean pooling.
        
        Args:
            sequences: List of protein sequences
            
        Returns:
            Mean-pooled embeddings
        """
        # Replace | with CLS CLS tokens
        processed_seqs = [s.replace('|', f"{self.cls_token}{self.cls_token}") for s in sequences]
        
        # Tokenize
        inputs = self.esm_tokenizer(processed_seqs, return_tensors="pt", padding=True, 
                                   truncation=True, max_length=1024)
        inputs = {k: v.to(self.esm_model.device) for k, v in inputs.items()}
        
        # Get embeddings
        outputs = self.esm_model(**inputs)
        hidden_states = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def forward(self, batch_input: Dict) -> Dict:
        """
        Args:
            batch_input: Dictionary with sequences and metadata
            
        Returns:
            Dictionary with predictions and metadata
        """
        protein_emb = self._get_esm_embeddings(batch_input["protein_sequence"])
        proteina_emb = self._get_esm_embeddings(batch_input["proteina_sequence"])
        
        labels = batch_input.get("labels")
        if labels is not None:
            labels = labels.to(self.esm_model.device)
        
        proj_output = self.projection_head(protein_emb, proteina_emb, labels)
        
        output = {
            "cosine_similarity": proj_output["cosine_similarity"],
            "original_pkds": batch_input["original_pkds"],
            "pdb_groups": batch_input["pdb_groups"],
            "subgroups": batch_input["subgroups"],
            "source_dataset": batch_input["source_dataset"]
        }
        
        if "loss" in proj_output:
            output["loss"] = proj_output["loss"]
        
        return output


class ProteinEmbeddingExtractor:
    """
    Utility class to prepare ESM-2 model with optional LoRA.
    """
    
    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D", 
                 device: str = "auto", lora_rank: int = 8, lora_alpha: int = 16, 
                 lora_dropout: float = 0.1, use_lora: bool = True):
        """
        Args:
            model_name: HuggingFace model identifier
            device: Device to use
            lora_rank: LoRA rank parameter
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout
            use_lora: Whether to apply LoRA
        """
        self.model_name = model_name
        self.use_lora = use_lora
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Initializing ProteinEmbeddingExtractor on {self.device}")
        
        self.dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        print(f"📥 Loading ESM-2 model: {model_name} with dtype: {self.dtype}")
        
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=self.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if use_lora:
            from peft import LoraConfig, get_peft_model, TaskType
            
            print(f"✨ Applying LoRA with rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=["key", "query", "value"]
            )
            self.model = get_peft_model(self.model, lora_config)
            print("   ✅ LoRA model loaded and configured. Trainable parameters:")
            self.model.print_trainable_parameters()
        else:
            print("   ⚠️ LoRA is disabled. All base model weights are frozen.")
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.to(self.device)
        self.embedding_size = self.model.config.hidden_size
        print(f"   ✅ Model loaded. Embedding size: {self.embedding_size}")
    
    def get_model_and_tokenizer(self) -> Tuple[nn.Module, object]:
        """
        Get the configured model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        return self.model, self.tokenizer
