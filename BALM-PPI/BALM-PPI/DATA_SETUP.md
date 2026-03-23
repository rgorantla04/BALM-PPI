# Dataset Setup Instructions

## Required Dataset

BALM-PPI requires the protein-protein binding affinity dataset: **PPB_Affinity_Sequences_Final (version 1).csv**

## File Location

Place the CSV file in the `data/` directory:

```
BALM-PPI/
└── data/
    └── PPB_Affinity_Sequences_Final (version 1).csv
```

## Dataset Format

The CSV file must contain the following columns:

### Required Columns

| Column Name | Type | Description | Example |
|---|---|---|---|
| `Target` | String | Target protein sequence | `MKFFL...VNQGG` |
| `proteina` | String | Query protein sequence | `MVHLT...FVQAAC` |
| `Y` | Float | pKd binding affinity value | `8.5` |
| `PDB` | String | PDB complex identifier | `1ABC` |
| `Subgroup` | String | Data subgroup label | `tight_binders` |
| `Source Data Set` | String | Source dataset identifier | `SKEMPI_v2` |

### Optional Columns (for BALM-PPI/PEFT)

| Column Name | Type | Description |
|---|---|---|
| `Ligand Name` | String | Ligand protein name |
| `Receptor Name` | String | Receptor protein name |

## Data Specifications

- **Protein Sequences**: 
  - Format: Single-letter amino acid codes
  - Separator for chain pairs: `|` (e.g., `SEQUENCE1|SEQUENCE2`)
  - No modifications required

- **pKd Values**:
  - Range: Typically 4-10 (binding affinity)
  - Format: Numeric float
  - Missing values: Automatically removed during loading

- **PDB Identifiers**:
  - Format: 4-letter code (e.g., `1ABC`, `3LZT`)
  - Used for cold target split
  - Cannot have NaN values

- **Subgroups**:
  - String labels for data categories
  - Examples: `tight_binders`, `weak_binders`, etc.

## Data Statistics

After loading, you should see:
```
✅ Data loaded and cleaned. Final dataset size: XXX samples.
pKd range: [Y_min: Z_min, Y_max: Z_max]
```

## Verification

Check your data with:

```python
import pandas as pd

# Load and verify
df = pd.read_csv('data/PPB_Affinity_Sequences_Final (version 1).csv')

# Check columns
print(df.columns.tolist())
# Expected: ['Target', 'proteina', 'Y', 'PDB', 'Subgroup', 'Source Data Set', ...]

# Check shape
print(f"Samples: {len(df)}")

# Check data types
print(df.dtypes)

# Check for missing values
print(df.isnull().sum())

# Check value ranges
print(f"pKd range: [{df['Y'].min():.2f}, {df['Y'].max():.2f}]")

# Show sample
print(df.head())
```

## Data Loading in Scripts

The data loader handles:
- ✅ Column selection and renaming
- ✅ Missing value removal
- ✅ Type conversion (Y to numeric)
- ✅ Data cleaning and validation

```python
from src.data.loader import load_dataset

df = load_dataset('data/PPB_Affinity_Sequences_Final (version 1).csv')
print(f"Loaded {len(df)} samples")
```

## Data Privacy & Citation

If using this dataset in publications, please:

1. **Cite Original Sources**:
   - SKEMPI database: Janin et al. (2003)
   - Other sources as indicated in 'Source Data Set' column

2. **Acknowledge Data Providers**:
   - Include in methods section
   - Provide dataset version information

3. **Data Availability**:
   - For reproducibility, ensure dataset can be obtained
   - Provide download instructions in supplement

## Troubleshooting

### FileNotFoundError: Dataset file not found
```
❌ ERROR: 'data.csv' not found. Please ensure the path is correct.
```

**Solution**:
```bash
# Check file exists
ls -la data/

# Verify exact filename
ls -la data/PPB*

# Update config if filename differs
```

### KeyError: Column not found
```
❌ ERROR: Column 'Target' not found in 'Data.csv'
```

**Solution**:
```bash
# Check available columns
python -c "import pandas as pd; df = pd.read_csv('data/PPB*.csv'); print(df.columns.tolist())"

# Update config with correct column names
```

### Type Conversion Issues
```
❌ ERROR: Cannot convert 'Y' column to numeric
```

**Solution**:
```python
# Check Y column values
import pandas as pd
df = pd.read_csv('data/PPB_Affinity_Sequences_Final (version 1).csv')
print(df['Y'].unique())
print(df['Y'].dtype)

# Remove non-numeric values before loading
```

## Data Requirements for Different Models

### Baseline Model
- ✅ Requires: Target, proteina, Y, all other columns optional
- ⏱️ Speed: Fast (embeddings pre-computed)
- 💾 Memory: Low

### Model-1
- ✅ Requires: Target, proteina, Y, PDB (for cold split)
- ⏱️ Speed: Fast (embeddings pre-computed)
- 💾 Memory: Low

### BALM-PPI (LoRA)
- ✅ Requires: Target, proteina, Y, PDB (for cold split)
- ⏱️ Speed: Slow (fine-tuning)
- 💾 Memory: High (GPU recommended)

### PLMs Ablation
- ✅ Requires: Same as Model-1
- ✅ Test different PLMs without data changes

## Dataset Size Recommendations

| Split Strategy | Minimum Samples | Recommended | Typical |
|---|---|---|---|
| Random | 50 | 200+ | 500-1000 |
| Cold Target | 100 | 300+ | 1000+ |
| Sequence Similarity | 200 | 400+ | 1000+ |

**Note**: Cold target and sequence similarity need more data for stable folds.

## Sample Dataset Creation

If you want to test the framework with a smaller dataset:

```python
import pandas as pd
import numpy as np

# Create minimal test dataset
sequences = ['MVHLTPEEKS', 'MKVLWAALLV', 'MKTAYIAKQR']
n_samples = 100

data = {
    'Target': np.random.choice(sequences, n_samples),
    'proteina': np.random.choice(sequences, n_samples),
    'Y': np.random.uniform(5, 10, n_samples),  # pKd range
    'PDB': [f'PDB{i%10}' for i in range(n_samples)],
    'Subgroup': np.random.choice(['group1', 'group2'], n_samples),
    'Source Data Set': ['TestData'] * n_samples,
    'Ligand Name': ['Ligand'] * n_samples,
    'Receptor Name': ['Receptor'] * n_samples,
}

df = pd.DataFrame(data)
df.to_csv('data/test_dataset.csv', index=False)
print("Test dataset created!")
```

Then configure in YAML:
```yaml
data:
  dataset_path: "data/test_dataset.csv"
```

## Data Augmentation & Preprocessing

Currently, no augmentation is applied. To add augmentation:

1. Modify `src/data/loader.py` → `ProteinPairEmbeddingDataset`
2. Add preprocessing functions
3. Apply in `__getitem__()` method

Examples:
- Sequence truncation
- Amino acid substitution
- Chain shuffling

## Benchmark Datasets

The framework can be adapted for other datasets:

- **SKEMPI**: Binding affinity mutations
- **PDBbind**: Comprehensive PDB dataset
- **CAPRI**: CASP-related targets
- **Custom**: Your own data

## Next Steps

1. ✅ Place dataset in `data/` directory
2. ✅ Verify with check script above
3. ✅ Run baseline test: `python train_baseline.py --config configs/baseline_config.yaml --split random`
4. ✅ Check results in `results/baseline/`

Good luck! 🧬
