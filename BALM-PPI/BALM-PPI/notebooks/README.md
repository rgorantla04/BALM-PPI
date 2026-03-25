# Usage

# Data Format
Your CSV file should contain the following columns:

Target: Target protein sequence

proteina: Query protein sequence

Y: pKd binding affinity value

PDB: PDB identifier (for cold split)

Subgroup: Data subgroup label

Source Data Set: Source dataset identifier

Ligand Name: Ligand name (for BALM-PPI)

Receptor Name: Receptor name (for BALM-PPI)

# Running Inference & Testing
To easily test our trained models, we provide a custom, user-friendly notebook for Batch Inference, Zero-Shot, and Few-Shot testing.

Navigate to the BALM-PPI/BALM-PPI/notebooks directory.

Open custom_notebook.ipynb.

Follow the interactive cells to load a pre-trained model (e.g., best_model_fold_1.pth) and pass in your custom protein sequences (CSV file with above format) to evaluate binding affinities without needing to run the full training pipelines.
