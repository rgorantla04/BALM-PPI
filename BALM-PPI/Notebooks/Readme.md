# Notebooks Directory

This directory contains inference and custom data notebooks for BALM-PPI.

## Contents

### `custom_notebook.ipynb` — Zero-Shot & Few-Shot Inference

A ready-to-use notebook for running BALM-PPI on your own protein-protein interaction data.

**Features:**
- **Zero-Shot Inference**: Predict binding affinity using the pretrained BALM-PPI model with no training required
- **Few-Shot Fine-Tuning**: Adapt the model to your data using a small labeled subset (30% by default)
- **Batch Processing**: Runs over a full CSV dataset efficiently
- **Automatic Weight Download**: Fetches pretrained weights from Hugging Face (`Harshit494/BALM-PPI`)
- **Evaluation & Plots**: Computes RMSE, Pearson, Spearman and generates regression plots

**Input format (CSV):**
| Column | Description |
|--------|-------------|
| `Target` | Target protein sequence (single-letter AA codes, `\|` for multi-chain) |
| `proteina` | Query protein sequence |
| `Y` | pKd binding affinity (optional, needed for evaluation) |

If no CSV is provided, a synthetic dummy dataset is generated automatically for testing.

**Output:**
- `Final_Test_Predictions.csv` — Test set predictions after few-shot fine-tuning
- Console metrics and regression comparison plots

### Data Files

- `Data.csv` — Sample dataset used with `custom_notebook.ipynb`
- `Final_Test_Predictions.csv` — Output predictions from the last notebook run
- `best_model_fold_1.pth` — Local copy of pretrained BALM-PPI weights (Fold 1)


