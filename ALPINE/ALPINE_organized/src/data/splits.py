"""
Data splitting strategies for cross-validation.
Includes: Random, Cold Target, and Sequence Similarity splitting.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import KFold, GroupKFold
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from tqdm import tqdm


def _get_sequence_similarity_splits(df: pd.DataFrame, n_folds: int = 5, 
                                   seed: int = 42) -> Tuple[List[Tuple], pd.DataFrame]:
    """
    Create k-folds based on sequence similarity using hierarchical clustering.
    
    Uses k-mer Jaccard similarity and agglomerative clustering to group similar sequences,
    then splits clusters into train/test folds.
    
    Args:
        df: DataFrame with 'Target' and 'proteina' columns containing sequences
        n_folds: Number of folds
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (list of (train_idx, test_idx), processed dataframe)
    """
    print("\n🧬 Performing k-mer/agglomerative sequence similarity splitting...")
    
    # CRITICAL: Work with a clean copy and reset index (matches notebook exactly)
    df = df.reset_index(drop=True).copy()

    # Get unique sequences with strip() for consistency (matches PEFT/notebook logic)
    all_sequences = set()
    sequence_to_records = {}
    for idx, row in df.iterrows():
        for col in ['Target', 'proteina']:
            seq = str(row[col]).strip()
            if seq:
                all_sequences.add(seq)
                if seq not in sequence_to_records:
                    sequence_to_records[seq] = []
                sequence_to_records[seq].append(idx)

    # CRITICAL: Sort for reproducibility
    unique_sequences = sorted(list(all_sequences))
    print(f"   Found {len(unique_sequences)} unique sorted sequences.")
    
    # Define k-mer function (matches notebook exactly, including short-sequence guard)
    def get_kmers(seq, k=3):
        if len(seq) < k:
            return {seq}
        return {seq[i:i+k] for i in range(len(seq) - k + 1)}
    
    # Get k-mers for all sequences
    kmer_sets = [get_kmers(s) for s in unique_sequences]

    # Calculate Jaccard similarity matrix (matches notebook loop-based approach)
    print("   Calculating sequence similarities...")
    n_seqs = len(unique_sequences)
    sim_matrix = np.zeros((n_seqs, n_seqs))
    for i in tqdm(range(n_seqs), desc="Similarity matrix", leave=False):
        for j in range(i, n_seqs):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                inter = len(kmer_sets[i] & kmer_sets[j])
                union = len(kmer_sets[i] | kmer_sets[j])
                sim = inter / union if union > 0 else 0.0
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

    # Convert to distance matrix and cluster
    print("   Clustering sequences...")
    distance_matrix = 1.0 - sim_matrix
    distance_threshold = 1.0 - 0.3  # Threshold = 0.3 similarity (matches notebook)

    condensed_dist = squareform(distance_matrix, checks=False)
    Z = linkage(condensed_dist, method='average')
    clusters = fcluster(Z, t=distance_threshold, criterion='distance')
    clusters = clusters - 1  # Convert to 0-indexed

    print(f"   Created {len(set(clusters))} clusters")

    # Map cluster IDs to record indices (matches notebook cluster_to_records logic)
    cluster_to_records = {}
    for seq_idx, cluster_id in enumerate(clusters):
        seq = unique_sequences[seq_idx]
        if cluster_id not in cluster_to_records:
            cluster_to_records[cluster_id] = []
        cluster_to_records[cluster_id].extend(sequence_to_records[seq])
    for cluster_id in cluster_to_records:
        cluster_to_records[cluster_id] = list(set(cluster_to_records[cluster_id]))

    # Create folds on clusters (matches notebook exactly)
    cluster_ids = list(cluster_to_records.keys())
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []

    for fold_idx, (train_cluster_idx, test_cluster_idx) in enumerate(kf.split(cluster_ids)):
        train_clusters = set([cluster_ids[i] for i in train_cluster_idx])
        test_clusters = set([cluster_ids[i] for i in test_cluster_idx])

        # Build sequence-to-group mapping (MATCHES PEFT/notebook logic exactly)
        sequence_to_group = {}
        for seq_idx, cluster_id in enumerate(clusters):
            sequence = unique_sequences[seq_idx]
            if cluster_id in train_clusters:
                sequence_to_group[sequence] = 'train'
            elif cluster_id in test_clusters:
                sequence_to_group[sequence] = 'test'

        # Assign rows to train/test (MATCHES PEFT logic: test if EITHER protein is in test)
        train_indices, test_indices = [], []
        for idx, row in df.iterrows():
            seq_target = str(row['Target']).strip()
            seq_proteina = str(row['proteina']).strip()
            cluster_target = sequence_to_group.get(seq_target)
            cluster_proteina = sequence_to_group.get(seq_proteina)
            if cluster_target == 'test' or cluster_proteina == 'test':
                test_indices.append(idx)
            else:
                train_indices.append(idx)

        train_idx = np.array(train_indices)
        test_idx = np.array(test_indices)

        print(f"   Fold {fold_idx + 1}: Train={len(train_idx)}, Test={len(test_idx)}")
        splits.append((train_idx, test_idx))

    return splits, df


def get_data_splits(df: pd.DataFrame, split_method: str = "random", 
                   n_folds: int = 5, seed: int = 42) -> Tuple[List[Tuple], pd.DataFrame]:
    """
    Get data splits according to specified method.
    
    Args:
        df: Input dataframe
        split_method: 'random', 'cold_target', or 'sequence_similarity'
        n_folds: Number of folds
        seed: Random seed
        
    Returns:
        Tuple of (list of splits, processed dataframe)
    """
    print(f"🔄 Creating {split_method} splits with {n_folds} folds")
    
    if split_method == "random":
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(kf.split(df))
        print(f"   ✅ Random splits created")
        return splits, df
    
    elif split_method == "cold_target":
        gkf = GroupKFold(n_splits=n_folds)
        groups = df["PDB"].astype(str).factorize()[0]  # Ensure string conversion for reproducibility
        splits = list(gkf.split(df, groups=groups))
        
        # Verify no leakage
        for fold, (train_idx, test_idx) in enumerate(splits):
            train_targets = set(df.iloc[train_idx]['PDB'].unique())
            test_targets = set(df.iloc[test_idx]['PDB'].unique())
            if not train_targets.intersection(test_targets):
                print(f"   ✅ Fold {fold+1}: No PDB leakage detected")
            else:
                print(f"   ⚠️ Fold {fold+1}: PDB leakage detected!")
        
        return splits, df
    
    elif split_method == "sequence_similarity":
        return _get_sequence_similarity_splits(df, n_folds, seed)
    
    else:
        raise ValueError(f"Unknown split method: {split_method}. "
                        f"Choose from: 'random', 'cold_target', 'sequence_similarity'")
