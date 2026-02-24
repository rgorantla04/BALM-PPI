"""
Data loading and preprocessing utilities for protein-protein binding affinity prediction.
"""

import os
import pickle
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Dict, List
from torch.utils.data import Dataset


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load and preprocess the protein-protein binding affinity dataset.
    
    Args:
        csv_path: Path to the CSV file containing the dataset
        
    Returns:
        Cleaned pandas DataFrame
    """
    try:
        df_raw = pd.read_csv(csv_path)
        
        # Select relevant columns
        df = df_raw[['Target', 'proteina', 'Y', 'PDB', 'Subgroup', 'Source Data Set']].copy()
        df.rename(columns={'Source Data Set': 'SourceDataSet'}, inplace=True)
        
        # Clean data
        df.dropna(subset=['Target', 'proteina', 'Y', 'PDB'], inplace=True)
        df['SourceDataSet'] = df['SourceDataSet'].astype(str)
        df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
        df.dropna(subset=['Y'], inplace=True)
        
        print(f"✅ Data loaded and cleaned. Final dataset size: {len(df)} samples.")
        print(f"   pKd range: [{df['Y'].min():.2f}, {df['Y'].max():.2f}]")
        
        df = df.reset_index(drop=True)  # Ensure index reset for reproducibility
        return df
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    except KeyError as e:
        raise KeyError(f"Required column {e} not found in dataset")


def get_pkd_bounds(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Get min and max pKd values for scaling.
    
    Args:
        df: DataFrame with 'Y' column containing pKd values
        
    Returns:
        Tuple of (min_pkd, max_pkd)
    """
    return float(df['Y'].min()), float(df['Y'].max())


class ProteinPairEmbeddingDataset(Dataset):
    """
    Dataset for protein pair embeddings.
    Works with pre-computed embeddings stored in a dictionary.
    """
    
    def __init__(self, dataframe: pd.DataFrame, embedding_dict: Dict, pkd_bounds: Tuple[float, float],
                 expected_embedding_size: int = None):
        """
        Args:
            dataframe: DataFrame with protein pairs and labels
            embedding_dict: Dictionary mapping sequence -> embedding tensor
            pkd_bounds: Tuple of (min_pkd, max_pkd) for scaling
        """
        self.data = dataframe.reset_index(drop=True)
        self.embedding_dict = embedding_dict
        self.pkd_lower, self.pkd_upper = pkd_bounds
        self.pkd_range = self.pkd_upper - self.pkd_lower
        # Expected embedding size (from embedding generator). If provided,
        # ensure returned embeddings are padded/truncated to this length.
        self.expected_embedding_size = expected_embedding_size
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        original_pkd = float(row['Y'])
        
        # Scale pKd to [-1, 1] range
        scaled_label = ((original_pkd - self.pkd_lower) / self.pkd_range) * 2 - 1
        
        def _fix_emb(emb):
            # Ensure embedding is a 1D torch.Tensor
            if isinstance(emb, np.ndarray):
                emb = torch.from_numpy(emb)
            emb = emb.detach() if isinstance(emb, torch.Tensor) else torch.tensor(emb)
            emb = emb.to(torch.float32)
            if emb.dim() == 2 and emb.size(0) == 1:
                emb = emb.squeeze(0)

            if self.expected_embedding_size is not None:
                exp = self.expected_embedding_size
                if emb.numel() < exp:
                    pad = torch.zeros(exp - emb.numel(), dtype=torch.float32)
                    emb = torch.cat([emb, pad], dim=0)
                elif emb.numel() > exp:
                    emb = emb[:exp]

            return emb

        return {
            "protein_embedding": _fix_emb(self.embedding_dict[row["Target"]]),
            "proteina_embedding": _fix_emb(self.embedding_dict[row["proteina"]]),
            "labels": torch.tensor(scaled_label, dtype=torch.float32),
            "original_pkds": torch.tensor(original_pkd, dtype=torch.float32),
            "pdb_groups": row["PDB"],
            "subgroups": row["Subgroup"],
            "source_dataset": row["SourceDataSet"]
        }


def collate_fn_embeddings(batch: List[dict]) -> dict:
    """
    Collate function for protein pair embedding datasets.
    
    Args:
        batch: List of batch items from the dataset
        
    Returns:
        Dictionary with stacked tensors and metadata lists
    """
    return {
        "protein_embedding": torch.stack([item['protein_embedding'] for item in batch]),
        "proteina_embedding": torch.stack([item['proteina_embedding'] for item in batch]),
        "labels": torch.stack([item['labels'] for item in batch]),
        "original_pkds": torch.stack([item['original_pkds'] for item in batch]),
        "pdb_groups": [item['pdb_groups'] for item in batch],
        "subgroups": [item['subgroups'] for item in batch],
        "source_dataset": [item['source_dataset'] for item in batch]
    }


class FastPPIDataset(Dataset):
    """
    Fast dataset for baseline model using pre-computed embeddings.
    """
    
    def __init__(self, dataframe: pd.DataFrame, protein1_embeddings: torch.Tensor, 
                 protein2_embeddings: torch.Tensor):
        """
        Args:
            dataframe: DataFrame with labels
            protein1_embeddings: Tensor of protein1 embeddings
            protein2_embeddings: Tensor of protein2 embeddings
        """
        self.data = dataframe.reset_index(drop=True)
        self.p1_emb = protein1_embeddings
        self.p2_emb = protein2_embeddings
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        return {
            "p1_emb": self.p1_emb[idx],
            "p2_emb": self.p2_emb[idx],
            "label": float(self.data.iloc[idx]["Y"])
        }


def fast_collate_fn(batch: List[dict]) -> dict:
    """
    Collate function for fast baseline datasets.
    
    Args:
        batch: List of batch items
        
    Returns:
        Dictionary with stacked tensors
    """
    return {
        "protein1_embeddings": torch.stack([item["p1_emb"] for item in batch]),
        "protein2_embeddings": torch.stack([item["p2_emb"] for item in batch]),
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.float32)
    }


def generate_and_cache_embeddings(df: pd.DataFrame, embedding_generator, 
                                  cache_path: str = "esm2_embeddings.pkl", 
                                  batch_size: int = 16) -> Tuple[Dict, int]:
    """
    Generate and cache embeddings for all unique protein sequences.
    
    Args:
        df: DataFrame with protein sequences
        embedding_generator: EmbeddingExtractor instance
        cache_path: Path to save/load embeddings cache
        batch_size: Batch size for embedding generation
        
    Returns:
        Tuple of (embedding_dict, embedding_size)
    """
    from tqdm import tqdm
    
    if os.path.exists(cache_path):
        print(f"📂 Loading pre-computed embeddings from '{cache_path}'...")
        with open(cache_path, 'rb') as f:
            embedding_dict, embedding_size = pickle.load(f)
        print(f"✅ Loaded {len(embedding_dict)} embeddings. Embedding size: {embedding_size}")
        return embedding_dict, embedding_size
    
    print("🛠️ No cache found. Generating new embeddings...")
    
    # Get unique sequences
    unique_seqs = pd.concat([df['Target'], df['proteina']]).unique()
    print(f"Found {len(unique_seqs)} unique protein sequences to process.")
    
    embedding_size = embedding_generator.embedding_size
    embedding_dict = {}
    
    # Generate embeddings in batches
    for i in tqdm(range(0, len(unique_seqs), batch_size), desc="Generating Embeddings"):
        batch_seqs = unique_seqs[i:i+batch_size]
        batch_embeddings = embedding_generator.get_embeddings(list(batch_seqs))
        
        for seq, emb in zip(batch_seqs, batch_embeddings):
            embedding_dict[seq] = emb
    
    # Save cache
    print(f"💾 Saving {len(embedding_dict)} embeddings to '{cache_path}'...")
    os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump((embedding_dict, embedding_size), f)
    
    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("✅ Embeddings generated and cached successfully.")
    return embedding_dict, embedding_size
