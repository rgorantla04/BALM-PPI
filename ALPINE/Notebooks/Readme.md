# Notebooks Directory

This directory contains the original Jupyter notebooks used during experimentation.
They are preserved here for reference and reproducibility purposes.

## Original Notebooks

### Baseline
- `BASELINE_NEW_CLS.ipynb` - Baseline model with frozen ESM-2

### Model-1 (Frozen ESM-2)
- `Model_1_Random.ipynb` - Model-1 with random split
- `Model_1_Cold.ipynb` - Model-1 with cold target split
- `Model_1_Sequence_Similarity.ipynb` - Model-1 with sequence similarity split

### ALPINE (LoRA Fine-tuning)
- `esm_2_peft_random.ipynb` - ALPINE with random split
- `esm2_peft_cold.ipynb` - ALPINE with cold target split
- `esm2_peft_seqsim.ipynb` - ALPINE with sequence similarity split

### PLMs Ablation Studies
- `ABLANG2_NEW_CLS.ipynb` - Ablang2 model with different projection sizes
- `ESM_2_CLS (256,512,1024).ipynb` - ESM-2 with different projection sizes
- `ESM_C_CLS.ipynb` - ESM-C model with different projection sizes
- `PROGEN_MEDIUM_CLS.ipynb` - PROGEN-2 Medium with different projection sizes
- `PROGEN_SMALL_NEW_CLS.ipynb` - PROGEN-2 Small with different projection sizes

## Note

The organized Python scripts in the parent directory replicate the functionality of these notebooks
with improved code organization, configuration management, and reproducibility. For new experiments,
please use the Python scripts rather than the notebooks.
