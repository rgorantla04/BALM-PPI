# BALM-PPI

A comprehensive framework for protein-protein binding affinity prediction using transformer-based protein language models (PLMs) with advanced training strategies.

## Overview

ALPINE provides three model architectures for predicting protein-protein binding affinity:

1. **Baseline Model**: Fast baseline using frozen ESM-2 embeddings with a simple projection head
2. **Model-1**: BALM architecture with frozen ESM-2 backbone and trainable projection head
3. **ALPINE**: Full BALM architecture with LoRA fine-tuning for efficient adaptation

Additionally, we include ablation studies comparing different protein language models (Ablang2, ESM-2, ESM-C, PROGEN-2).

## Project Structure

```
ALPINE_organized/
├── src/
│   ├── models/              # Model architectures and training utilities
│   │   ├── architectures.py # Model definitions
│   │   ├── training.py      # Training and evaluation functions
│   │   └── __init__.py
│   ├── data/                # Data loading and processing
│   │   ├── loader.py        # Dataset classes and loading functions
│   │   ├── embeddings.py    # Embedding extraction from PLMs
│   │   ├── splits.py        # CV splitting strategies
│   │   └── __init__.py
│   └── utils/               # Utility functions
│       ├── reproducibility.py # Random seed setup
│       ├── metrics.py        # Evaluation metrics
│       ├── config.py         # Configuration loading
│       ├── visualization.py  # Plotting functions
│       └── __init__.py
├── configs/                 # Configuration files
│   ├── baseline_config.yaml # Baseline model config
│   ├── model_1_config.yaml  # Model-1 config
│   ├── alpine_config.yaml   # ALPINE config
│   └── plms_config.yaml     # PLMs ablation config
├── data/                    # Dataset (add your CSV file here)
├── cache/                   # Cached embeddings
├── results/                 # Training results and predictions
├── notebooks/               # Original Jupyter notebooks (for reference)
├── train_baseline.py        # Baseline training script
├── train_model1.py          # Model-1 training script
├── train_alpine.py          # ALPINE training script
├── train_plms.py            # PLMs ablation script
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── .gitignore              # Git ignore file
```

## Installation

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (optional, for GPU support)

### Setup

1. Clone the repository:
```bash
cd ALPINE_organized
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place your dataset in the `data/` directory:
```bash
cp "PPB_Affinity_Sequences_Final (version 1).csv" data/
```

## Usage

### Data Format

Your CSV file should contain the following columns:
- `Target`: Target protein sequence
- `proteina`: Query protein sequence
- `Y`: pKd binding affinity value
- `PDB`: PDB identifier (for cold split)
- `Subgroup`: Data subgroup label
- `Source Data Set`: Source dataset identifier
- `Ligand Name`: Ligand name (for ALPINE)
- `Receptor Name`: Receptor name (for ALPINE)

### Running Experiments

#### 1. Baseline Model

```bash
# Random split
python train_baseline.py --config configs/baseline_config.yaml --split random

# Cold target split
python train_baseline.py --config configs/baseline_config.yaml --split cold_target

# Sequence similarity split
python train_baseline.py --config configs/baseline_config.yaml --split sequence_similarity
```

#### 2. Model-1 (Frozen ESM-2)

```bash
# Cold target split (main configuration)
python train_model1.py --config configs/model_1_config.yaml --split cold_target
```

#### 3. ALPINE (LoRA Fine-tuning)

```bash
# Cold target split with LoRA
python train_alpine.py --config configs/alpine_config.yaml --split cold_target
```

#### 4. PLMs Ablation Study

```bash
# Test all PLMs with Model-1 architecture
python train_plms.py --config configs/plms_config.yaml --plm esm2
python train_plms.py --config configs/plms_config.yaml --plm ablang2
```

### Configuration

All experiments are configured via YAML files in the `configs/` directory. Key parameters:

- **Data**: Dataset path, cache locations
- **Model**: PLM selection, projection sizes, LoRA settings
- **Training**: Learning rate, epochs, batch size, patience
- **Device**: GPU/CPU selection, mixed precision
- **Output**: Results directory, saving options

Edit configuration files to customize experiments without modifying code.

## Key Features

### Data Splitting Strategies

1. **Random**: Standard random k-fold splitting
2. **Cold Target**: Group-k-fold based on PDB identifiers (cold start)
3. **Sequence Similarity**: Hierarchical clustering based on k-mer Jaccard similarity

### Reproducibility

All experiments use fixed random seeds and deterministic operations:
```python
from src.utils.reproducibility import setup_reproducibility
setup_reproducibility(seed=42)
```

### Evaluation Metrics

- **RMSE**: Root Mean Square Error
- **Pearson**: Pearson correlation coefficient
- **Spearman**: Spearman rank correlation
- **CI**: Concordance Index (for ranking)

### Model Architectures

**BALMProjectionHead**: 
- Separate projection layers for each protein
- L2 normalization
- Cosine similarity computation
- MSE loss

**Baseline & Model-1**:
- Frozen transformer backbone
- Fast inference (pre-computed embeddings)
- Low memory footprint

**ALPINE**:
- LoRA fine-tuning on transformer attention layers
- Efficient parameter adaptation
- Better performance for domain-specific tasks

## Training Tips

1. **Memory**: Use larger batch sizes for baseline (batch_size=16-32)
2. **LoRA**: Recommended rank=8, alpha=16 for balance
3. **Learning Rate**: 1e-4 to 1e-3 depending on model and dataset
4. **Patience**: Early stopping after 15 epochs without improvement

## Results

Results are saved in `results/{model_name}/`:
- `fold_*_predictions.csv`: Per-fold predictions
- `cv_summary_metrics.csv`: Cross-validation summary
- `overall_regression.png`: Regression plot
- `model_metrics.json`: Detailed metrics

## Citation

If you use ALPINE in your research, please cite:

```bibtex
@article{alpine2024,
  title={ALPINE: Advanced Learning on Protein-Protein Interaction Networks},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

[Add your license here]

## Support

For issues or questions:
1. Check the configuration files for correct settings
2. Verify dataset format matches expected structure
3. Ensure CUDA/GPU is available if using GPU mode
4. Check cache directory has write permissions

## Original Notebooks

Original Jupyter notebooks are preserved in the `notebooks/` directory for reference:
- `BASELINE_NEW_CLS.ipynb`: Baseline model implementation
- `Model_1_*.ipynb`: Model-1 variants (3 splits)
- `esm2_peft_*.ipynb`: ALPINE with PEFT/LoRA (3 splits)
- `*_CLS.ipynb`: PLMs ablation studies

## Changelog

### v1.0.0 (2024)
- Initial release
- Baseline model
- Model-1 architecture
- ALPINE with LoRA
- PLMs ablation studies
- Comprehensive configuration system
- Full reproducibility support
