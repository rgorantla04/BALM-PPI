# BALM-PPI: Advanced Learning on Protein-Protein Interaction Networks

A comprehensive framework for protein-protein binding affinity prediction using transformer-based protein language models (PLMs) with advanced training strategies.

## Overview

BALM-PPI provides three model architectures for predicting protein-protein binding affinity:

![alt text](architecture.png)
1. **Baseline Model**: Fast baseline using frozen ESM-2 embeddings with a simple projection head
2. **Model-1**: BALM architecture with frozen ESM-2 backbone and trainable projection head
3. **BALM-PPI**: Full BALM architecture with LoRA fine-tuning for efficient adaptation

Additionally, we include ablation studies comparing different protein language models (Ablang2, ESM-2, ESM-C, PROGEN-2).


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

### Option 1: Web Tool (No Setup Required)

The easiest way to use BALM-PPI — paste your FASTA sequences directly and get binding affinity predictions instantly, no installation needed:

**[BALM-PPI Web Tool on Hugging Face Spaces](https://huggingface.co/spaces/Harshit494/BALM-PPI)**

Supports standard FASTA format for both target and query proteins.

---

### Option 2: Inference Notebook (Custom Data, Zero/Few-Shot)

Use `notebooks/custom_notebook.ipynb` to run inference on your own dataset locally.

**What it does:**
- **Zero-Shot**: Loads pretrained BALM-PPI weights from Hugging Face automatically and predicts binding affinity on your data — no training required
- **Few-Shot Fine-Tuning**: Adapts the model to your specific data using a small labeled subset (30% by default, configurable)
- **Batch Processing**: Runs over a full CSV file with progress tracking
- **Evaluation**: Computes RMSE, Pearson, Spearman and plots regression comparisons between zero-shot and few-shot results

**Input CSV format:**

| Column | Description |
|--------|-------------|
| `Target` | Target protein sequence (single-letter AA codes; use `\|` to separate chains) |
| `proteina` | Query/ligand protein sequence |
| `Y` | pKd value (optional — only needed for evaluation metrics) |

If no CSV is provided, a synthetic dummy dataset is generated automatically so you can test the notebook immediately.

**Steps to run:**
1. Open `notebooks/custom_notebook.ipynb` in Jupyter
2. Place your CSV in the `notebooks/` directory (or update `CSV_PATH` in the notebook)
3. Adjust `PKD_LOWER_BOUND` / `PKD_UPPER_BOUND` to match your training data's pKd range
4. Run all cells — pretrained weights download automatically
5. Results are saved to `notebooks/Final_Test_Predictions.csv`



### Option 3: Full Training from Scratch

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

**Baseline & Model-1**:
- Frozen transformer backbone
- Fast inference (pre-computed embeddings)
- Low memory footprint

**BALM-PPI**:
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

If you use BALM-PPI in your research, please cite:

```bibtex
@article{balm_ppi2024,
  title={BALM-PPI: Advanced Learning on Protein-Protein Interaction Networks},
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

Original Jupyter notebooks are preserved in `Training_notebooks/` for reference, organized by model:

- `Training_notebooks/Baseline Notebooks/BASELINE_NEW_CLS (1).ipynb` — Baseline model
- `Training_notebooks/BALM-PPI without PEFT Notebooks/` — Model-1 (random, cold, seqsim splits)
- `Training_notebooks/BALM-PPI(PEFT) Notebooks/` — BALM-PPI with LoRA (random, cold, seqsim splits)
- `Training_notebooks/PLMs Notebooks/` — Ablang2, ESM-2, ESM-C, ProGen ablation studies

## Changelog

### v1.0.0 (2024)
- Initial release
- Baseline model
- Model-1 architecture
- BALM-PPI with LoRA
- PLMs ablation studies
- Comprehensive configuration system
- Full reproducibility support
