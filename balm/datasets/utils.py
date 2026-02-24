from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from tdc.multi_pred import DTI


from balm.configs import DatasetConfigs, TrainingConfigs


# In balm/datasets/utils.py

# Add ESM import
import esm
import torch
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any

@dataclass
class ESMDataCollator:
    """
    Data collator for ESM-2 that handles batching protein sequences.

    Attributes:
        batch_converter: The ESM-2 batch converter for processing sequences.
        protein_max_length: Maximum allowed length for first protein sequences.
        proteina_max_length: Maximum allowed length for second protein sequences.
    """
    batch_converter: Any
    protein_max_length: int
    proteina_max_length: int

    def __call__(self, features: List[Dict[str, Union[str, float]]]) -> Dict[str, Any]:
        """
        Convert a list of sequences to ESM-2 batch format.

        Args:
            features (List[Dict]): List of dictionaries containing protein sequences and labels.

        Returns:
            Dict: Processed batch with tokens and labels.
        """
        protein_sequences = [f["protein"] for f in features]
        proteina_sequences = [f["proteina"] for f in features]
        labels = [f["Y"] for f in features]
        
        # Process protein sequences
        protein_data = [(str(i), seq) for i, seq in enumerate(protein_sequences)]
        _, _, protein_tokens = self.batch_converter(protein_data)
        
        # Process proteina sequences
        proteina_data = [(str(i), seq) for i, seq in enumerate(proteina_sequences)]
        _, _, proteina_tokens = self.batch_converter(proteina_data)
        
        # Create the batch dictionary
        batch = {
            "protein_sequences": protein_sequences,
            "proteina_sequences": proteina_sequences,
            "protein_tokens": protein_tokens,
            "proteina_tokens": proteina_tokens,
            "labels": torch.tensor(labels, dtype=torch.float32)
        }
        
        # Store original sequences for metrics computation
        batch["protein_ori_sequences"] = protein_sequences
        batch["proteina_ori_sequences"] = proteina_sequences
        
        return batch
        
class DataCollatorWithPadding:
    def __init__(
        self,
        protein_tokenizer: PreTrainedTokenizerBase,
        proteina_tokenizer: PreTrainedTokenizerBase,
        padding: Union[bool, str, PaddingStrategy] = True,
        protein_max_length: Optional[int] = None,
        proteina_max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
    ):
        self.protein_tokenizer = protein_tokenizer
        self.proteina_tokenizer = proteina_tokenizer
        self.padding = padding
        self.protein_max_length = protein_max_length
        self.proteina_max_length = proteina_max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract 'protein_input_ids' and prepare them for padding
        protein_features = [
            {"input_ids": feature["protein_input_ids"]} for feature in features
        ]

        # Pad 'protein_input_ids' and ensure they're named correctly after padding
        padded_protein_features = self.protein_tokenizer.pad(
            protein_features,
            padding=self.padding,
            max_length=self.protein_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # Extract 'proteina_input_ids' and prepare them for padding
        proteina_features = [
            {"input_ids": feature["proteina_input_ids"]} for feature in features
        ]

        # Pad 'proteina_input_ids' and ensure they're named correctly after padding
        padded_proteina_features = self.proteina_tokenizer.pad(
            proteina_features,
            padding=self.padding,
            max_length=self.proteina_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch = {
            "protein_ori_sequences": [
                feature["protein_ori_sequences"] for feature in features
            ],
            "proteina_ori_sequences": [
                feature["proteina_ori_sequences"] for feature in features
            ],
            "protein_input_ids": padded_protein_features["input_ids"],
            "protein_attention_mask": padded_protein_features["attention_mask"],
            "proteina_input_ids": padded_proteina_features["input_ids"],
            "proteina_attention_mask": padded_proteina_features["attention_mask"],
            "labels": torch.stack([torch.tensor(feature["Y"]) for feature in features]),
        }

        return batch

