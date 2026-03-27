# Training Notebooks

This directory contains the original Jupyter notebooks used during experimentation.
They are preserved here for reference and reproducibility purposes.

For new experiments, use the Python training scripts (`train_*.py`) with YAML configs in `configs/`.

## Structure

### Baseline Notebooks/
- `BASELINE_NEW_CLS (1).ipynb` — Baseline model with frozen ESM-2 embeddings

### BALM-PPI without PEFT Notebooks/ (Model-1, Frozen ESM-2)
- `Model_1_Random.ipynb` — Model-1 with random split
- `Model_1_Cold.ipynb` — Model-1 with cold target split
- `Model_1_seqsim.ipynb` — Model-1 with sequence similarity split

### BALM-PPI(PEFT) Notebooks/ (LoRA Fine-tuning)
- `esm_2_peft_random.ipynb` — BALM-PPI with random split
- `esm2_peft_cold.ipynb` — BALM-PPI with cold target split
- `esm2_peft_seqsim.ipynb` — BALM-PPI with sequence similarity split

### PLMs Notebooks/ (Ablation Studies)
- `ABLANG2_NEW_CLS.ipynb` — Ablang2 with different projection sizes
- `ESM_2_CLS (256,512,1024).ipynb` — ESM-2 with different projection sizes
- `ESM_C_CLS.ipynb` — ESM-C model with different projection sizes
- `PROGEN_MEDIUM_CLS.ipynb` — ProGen-2 Medium with different projection sizes
- `PROGEN_SMALL_NEW_CLS.ipynb` — ProGen-2 Small with different projection sizes

## Note

The Python modules in `src/` replicate the functionality of these notebooks with improved
code organization, configuration management, and reproducibility. For inference on custom
data, see `notebooks/custom_notebook.ipynb`.
