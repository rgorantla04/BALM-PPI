# BALM-PPI: Quick Start Guide

This guide will help you get BALM-PPI up and running quickly.

## 1. Environment Setup (5 minutes)

### Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 2. Prepare Data (2 minutes)

1. Get your dataset CSV file: `PPB_Affinity_Sequences_Final (version 1).csv`
2. Place it in the `data/` directory
3. Verify columns: Target, proteina, Y, PDB, Subgroup, Source Data Set

```bash
ls -la data/
```

## 3. Run a Quick Experiment (10-30 minutes depending on hardware)

### Option A: Baseline Model (Fastest)
```bash
# Random split (test everything works)
python train_baseline.py --config configs/baseline_config.yaml --split random
```

Expected output:
- Results in `results/baseline/`
- Predictions CSV and metrics
- Regression plot

### Option B: Model-1 Cold Split (Our Main Model)
```bash
python train_model1.py --config configs/model_1_config.yaml --split cold_target
```

### Option C: BALM-PPI with LoRA (Best Performance but Slower)
```bash
python train_balm_ppi.py --config configs/balm_ppi_config.yaml --split cold_target
```

## 4. Customize Configuration

Edit configuration files to change parameters without modifying code:

```yaml
# Example: configs/model_1_config.yaml
training:
  epochs: 50          # More epochs
  batch_size: 32      # Larger batch
  learning_rate: 5e-4 # Different learning rate
  patience: 20        # More patience before early stopping
```

Common parameters to adjust:
- `learning_rate`: 1e-5 to 1e-3
- `batch_size`: 1 to 32 (depends on GPU memory)
- `epochs`: 20 to 100
- `patience`: 10 to 30

## 5. Reproduce All Three Splits

Run all three splitting strategies to fully replicate our experiments:

```bash
# Model-1 with all splits
python train_model1.py --config configs/model_1_config.yaml --split random
python train_model1.py --config configs/model_1_config.yaml --split cold_target
python train_model1.py --config configs/model_1_config.yaml --split sequence_similarity

# BALM-PPI with all splits
python train_balm_ppi.py --config configs/balm_ppi_config.yaml --split random
python train_balm_ppi.py --config configs/balm_ppi_config.yaml --split cold_target
python train_balm_ppi.py --config configs/balm_ppi_config.yaml --split sequence_similarity
```

## 6. Understanding Results

After running an experiment, check `results/{model_name}/`:

```
results/
├── fold_1_predictions.csv        # Fold 1 predictions
├── fold_2_predictions.csv        # Fold 2 predictions
├── ...
├── fold_5_predictions.csv        # Fold 5 predictions
├── cv_summary_metrics.csv        # Cross-validation summary
└── overall_regression.png        # Regression visualization
```

### Metrics in CSV:
- `label`: True pKd value
- `prediction`: Predicted pKd
- `residual`: (true - predicted)
- `abs_residual`: |true - predicted|
- `PDB`: PDB identifier
- `Subgroup`: Data subgroup
- `Source Data Set`: Source dataset

### Summary Metrics:
- **RMSE**: Root Mean Square Error (lower is better)
- **Pearson**: Pearson correlation (higher is better)
- **Spearman**: Spearman correlation (higher is better)
- **CI**: Concordance Index (higher is better)

## 7. GPU Memory Tips

If you get CUDA out of memory errors:

```python
# In config file, reduce:
batch_size: 8  # Instead of 16
```

Or use CPU:
```yaml
device:
  type: "cpu"
```

## 8. Next Steps

1. **Modify Dataset**: Update `data.dataset_path` in config
2. **Change PLM**: Update `model.model_name` (e.g., "AbLang/ablang2-base")
3. **Adjust Architecture**: Change `model.projected_size` (256, 512, 1024)
4. **Implement Custom**: Edit `src/models/architectures.py`

## 9. Troubleshooting

### ImportError: No module named 'peft'
```bash
pip install peft
```

### CUDA out of memory
- Reduce batch_size in config
- Use CPU: set `device.type: cpu`
- Clear cache: `torch.cuda.empty_cache()`

### Embedding cache issue
Delete `cache/` and re-run to regenerate:
```bash
rm -rf cache/
```

### Dataset not found
Ensure CSV is in `data/` directory and path matches config:
```bash
ls data/PPB_Affinity_Sequences_Final*
```

## 10. Reading the Code

Key files to understand:
- `src/models/architectures.py` - Model definitions
- `src/data/loader.py` - Data loading and datasets
- `src/data/splits.py` - Data splitting strategies
- `train_model1.py` - Main training loop example

## Useful Commands

```bash
# Check GPU
nvidia-smi

# Monitor training (in separate terminal)
watch -n 1 nvidia-smi

# View dataset
python -c "import pandas as pd; df = pd.read_csv('data/PPB_Affinity_Sequences_Final (version 1).csv'); print(df.head()); print(df.info())"

# Check config syntax
python -c "from src.utils.config import load_config; config = load_config('configs/model_1_config.yaml'); print(config['model'])"
```

## Performance Expectations

Typical results (5-fold CV, cold target split):

| Model | RMSE | Pearson | Spearman | CI |
|-------|------|---------|----------|-----|
| Baseline | 0.95 | 0.72 | 0.70 | 0.75 |
| Model-1 | 0.92 | 0.75 | 0.73 | 0.77 |
| BALM-PPI | 0.89 | 0.77 | 0.75 | 0.78 |

*Note: These are example values. Your results will depend on your dataset.*

## Need Help?

1. Check `README.md` for detailed documentation
2. Review original notebooks in `notebooks/` for reference
3. Check YAML config files for all available options
4. Read docstrings in Python files: `python -c "import src.models.architectures; help(src.models.architectures.BALMForRegression)"`

Good luck! 🧬
