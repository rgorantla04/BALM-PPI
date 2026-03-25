# BALM-PPI: Advanced Learning on Protein-Protein Interaction Networks

A comprehensive framework for protein-protein binding affinity prediction using transformer-based protein language models (PLMs) with advanced training strategies.

## Overview

BALM-PPI provides three model architectures for predicting protein-protein binding affinity:

1. **Baseline Model**: Fast baseline using frozen ESM-2 embeddings with a simple projection head
2. **Model-1**: BALM architecture with frozen ESM-2 backbone and trainable projection head
3. **BALM-PPI**: Full BALM architecture with LoRA fine-tuning for efficient adaptation

Additionally, we include ablation studies comparing different protein language models (Ablang2, ESM-2, ESM-C, PROGEN-2).

## Project Structure

```
BALM-PPI/
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
│   ├── balm_ppi_config.yaml   # BALM-PPI config
│   └── plms_config.yaml     # PLMs ablation config
├── data/                    # Dataset (add your CSV file here)
├── cache/                   # Cached embeddings
├── results/                 # Training results and predictions
├── notebooks/               # Original Jupyter notebooks (for reference)
├── train_baseline.py        # Baseline training script
├── train_model1.py          # Model-1 training script
├── train_balm_ppi.py          # BALM-PPI training script
├── train_plms.py            # PLMs ablation script
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── .gitignore              # Git ignore file
```

### Setup

1. Clone the repository:
```bash
cd BALM-PPI
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
cp "PPB_Affinity_Sequences.csv" data/
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
- `Ligand Name`: Ligand name (for BALM-PPI)
- `Receptor Name`: Receptor name (for BALM-PPI)

# Running Inference & Testing on custom dataset

To easily test our trained models, we provide a custom, user-friendly notebook for Batch Inference, Zero-Shot, and Few-Shot testing.

Navigate to the notebooks/ directory.

Open custom_notebook.ipynb.

Follow the interactive cells to load a pre-trained model (e.g., best_model_fold_1.pth) and pass in your custom protein sequences to evaluate binding affinities without needing to run the full training pipelines.


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

#### 3. BALM-PPI (LoRA Fine-tuning)

```bash
# Cold target split with LoRA
python train_balm_ppi.py --config configs/balm_ppi_config.yaml --split cold_target
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

**Baseline & Model-1 (BALM-PPI without PEFT)**:
- Frozen transformer backbone
- Fast inference (pre-computed embeddings)
- Low memory footprint

**BALM-PPI**:
- LoRA fine-tuning on transformer attention layers
- Efficient parameter adaptation
- Better performance for domain-specific tasks



## Results

Results are saved in `results/{model_name}/`:
- `fold_*_predictions.csv`: Per-fold predictions
- `cv_summary_metrics.csv`: Cross-validation summary
- `overall_regression.png`: Regression plot
- `model_metrics.json`: Detailed metrics

## Citation

If you use BALM-PPI in your research, please cite:



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
- `esm2_peft_*.ipynb`: BALM-PPI with PEFT/LoRA (3 splits)
- `*_CLS.ipynb`: PLMs ablation studies


