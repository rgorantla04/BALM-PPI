"""
Real, measured wall-clock inference timing for BALM-PPI, to replace the
approximate "documented web-server timing" previously cited in the
computational-efficiency comparison table (R2-8).

Loads the real, HuggingFace-verified checkpoint (cv_results_lora_random_target_new)
and times single-complex forward passes (no gradients, eval mode), on both GPU
and CPU, over a sample of real complexes spanning a range of sequence lengths
drawn directly from the real dataset -- same spirit as the PPB-Affinity timing
(measured over many real structures), not a synthetic benchmark.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_multiseed_cv import build_model  # noqa: E402

CKPT_PATH = "cv_results_lora_random_target_new/best_model_fold_1.pth"
DATA_CSV = "notebooks/Data.csv"
N_SAMPLES = 15
N_WARMUP = 3


def load_model(device):
    model = build_model(device, (1.0, 16.0))
    state = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing + unexpected if 'position_embeddings' not in k]
    assert not bad, f"Unexpected state_dict mismatch: {bad}"
    model.eval()
    return model


def time_inference(model, device, rows):
    times = []
    with torch.no_grad():
        for i, row in enumerate(rows):
            batch = {
                "protein_sequence": [row["Target"]],
                "proteina_sequence": [row["proteina"]],
                "original_pkds": [row["Y"]],
                "pdb_groups": [row.get("PDB", "")],
                "subgroups": [row.get("Subgroup", "")],
            }
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            if i >= N_WARMUP:  # discard warmup iterations (model/cudnn warmup)
                times.append(t1 - t0)
    return times


def main():
    df = pd.read_csv(DATA_CSV).dropna(subset=["Target", "proteina", "Y"]).reset_index(drop=True)
    # Sample across the length distribution, not just the first N rows.
    df["total_len"] = df["Target"].str.len() + df["proteina"].str.len()
    df_sorted = df.sort_values("total_len")
    idxs = np.linspace(0, len(df_sorted) - 1, N_SAMPLES + N_WARMUP, dtype=int)
    rows = [df_sorted.iloc[i] for i in idxs]
    lens = [r["total_len"] for r in rows]
    print(f"Sampled {len(rows)} complexes, total-length range {min(lens)}-{max(lens)} residues")

    results = {}
    for device_str in ["cuda", "cpu"]:
        if device_str == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, skipping GPU timing")
            continue
        device = torch.device(device_str)
        print(f"\n=== {device_str.upper()} ===")
        print("Loading model...")
        model = load_model(device)
        times = time_inference(model, device, rows)
        arr = np.array(times)
        print(f"Per-complex inference time ({device_str}): "
              f"mean={arr.mean():.3f}s  std={arr.std():.3f}s  "
              f"min={arr.min():.3f}s  max={arr.max():.3f}s  (n={len(arr)})")
        results[device_str] = arr
        del model
        if device_str == "cuda":
            torch.cuda.empty_cache()

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"{k}: {v.mean():.2f} +/- {v.std():.2f} s/complex")


if __name__ == "__main__":
    main()
