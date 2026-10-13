"""
Quantitative hotspot-recovery summary for the rebuttal (Editor E3 / R2-4 / R3-3).

Pulls the already-computed per-complex, per-protein attribution-vs-ground-truth
metrics from paper_results/complete_validation_results.csv (13 complexes,
26 protein sides), and the protein length / hotspot-count figures printed in
"esm_explain all with GT.ipynb" (ground-truth hotspot definitions per protein),
to report:
  - Pooled mean +/- s.d. AUROC, AUPRC across all 26 protein sides.
  - Pooled mean +/- s.d. Precision/Recall/F1 at the fixed 25% attribution
    threshold (chosen because it is the threshold used for the majority of
    complexes in P1_Best_Thresh/P2_Best_Thresh, and because reporting a FIXED
    threshold avoids the optimistic bias of per-protein threshold selection
    that P1_Best_F1/P2_Best_F1 in the source CSV would introduce).
  - The analytically-correct RANDOM BASELINE for the same fixed threshold:
    for a random top-25% residue selection, E[Recall] = 0.25 exactly
    (independent of population size, by symmetry of simple random sampling
    without replacement), and E[Precision] = h/n (the true hotspot base rate
    for that protein), so E[F1] = 2*0.25*(h/n) / (0.25 + h/n). This baseline
    is computed per protein side using its own (n, h) and then pooled the
    same way as the observed metrics, giving a fair per-complex comparison
    rather than one global random number.
"""
import csv
import statistics as stats
from pathlib import Path

CSV_PATH = Path(__file__).parent / "paper_results" / "complete_validation_results.csv"

# (protein length n, hotspot count h) per complex per side, transcribed from
# printed ground-truth summaries in esm_explain all with GT.ipynb (verified
# against the same complexes/PDB IDs as complete_validation_results.csv).
GROUND_TRUTH_COUNTS = {
    "HyHEL-Lysozyme": {"P1": (129, 4), "P2": (429, 6)},
    "Lysozyme-HyHEL5": {"P1": (129, 3), "P2": (426, 7)},
    "Lysozyme-D1.3": {"P1": (129, 1), "P2": (223, 6)},
    "HER2-Trastuzumab": {"P1": (607, 20), "P2": (227, 9)},
    "HER2-Pertuzumab": {"P1": (624, 3), "P2": (231, 10)},
    "PD1-Pembrolizumab": {"P1": (114, 8), "P2": (436, 17)},
    "TNF-Infliximab": {"P1": (157, 3), "P2": (442, 7)},
    "Barnase-Barstar": {"P1": (110, 5), "P2": (89, 8)},
    "Lysozyme-Nanobody": {"P1": (129, 3), "P2": (148, 8)},
    "Trypsin-BPTI": {"P1": (223, 6), "P2": (58, 4)},
    "MDM2-p53": {"P1": (109, 10), "P2": (15, 3)},
    "Ras-Raf": {"P1": (170, 7), "P2": (137, 0)},
    "Colicin-Immunity": {"P1": (134, 8), "P2": (86, 11)},
}

THRESH = "25"  # fixed threshold used for the headline comparison


def random_baseline(n: int, h: int, k: float = 0.25):
    if h == 0:
        return 0.0, 0.0, 0.0  # no hotspots to recover -- P/R/F1 baseline is degenerate 0
    precision = h / n
    recall = k
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main():
    auroc_vals, auprc_vals = [], []
    obs_p, obs_r, obs_f1 = [], [], []
    rand_p, rand_r, rand_f1 = [], [], []
    skipped_zero_hotspot = []

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            complex_name = row["Complex"]
            gt = GROUND_TRUTH_COUNTS[complex_name]
            for side in ["P1", "P2"]:
                n, h = gt[side]
                auroc_vals.append(float(row[f"{side}_AUROC"]))
                auprc_vals.append(float(row[f"{side}_AUPRC"]))

                p_obs = float(row[f"{side}_Precision@{THRESH}%"])
                r_obs = float(row[f"{side}_Recall@{THRESH}%"])
                f1_obs = float(row[f"{side}_F1@{THRESH}%"])

                p_rand, r_rand, f1_rand = random_baseline(n, h, k=float(THRESH) / 100)

                if h == 0:
                    skipped_zero_hotspot.append(f"{complex_name}/{side}")
                    continue  # exclude degenerate no-hotspot sides from pooled P/R/F1 stats

                obs_p.append(p_obs)
                obs_r.append(r_obs)
                obs_f1.append(f1_obs)
                rand_p.append(p_rand)
                rand_r.append(r_rand)
                rand_f1.append(f1_rand)

    def mean_sd(xs):
        return stats.mean(xs), stats.pstdev(xs)

    n_auroc = len(auroc_vals)
    n_pr = len(obs_p)

    print(f"Pooled over {n_auroc} protein sides (13 complexes x 2 sides):")
    m, s = mean_sd(auroc_vals)
    print(f"  AUROC : {m:.3f} +/- {s:.3f}   (random baseline = 0.500 by definition)")
    m, s = mean_sd(auprc_vals)
    print(f"  AUPRC : {m:.3f} +/- {s:.3f}")

    print(f"\nPooled over {n_pr} protein sides with >=1 true hotspot (excluded: {skipped_zero_hotspot}), at fixed {THRESH}% attribution threshold:")
    m, s = mean_sd(obs_p)
    rm, rs = mean_sd(rand_p)
    print(f"  Precision@{THRESH}% : observed {m:.3f} +/- {s:.3f}   vs random {rm:.3f} +/- {rs:.3f}")
    m, s = mean_sd(obs_r)
    rm, rs = mean_sd(rand_r)
    print(f"  Recall@{THRESH}%    : observed {m:.3f} +/- {s:.3f}   vs random {rm:.3f} +/- {rs:.3f}  (random Recall is exactly {THRESH}%/100 = {float(THRESH)/100:.2f} by construction for every side)")
    m, s = mean_sd(obs_f1)
    rm, rs = mean_sd(rand_f1)
    print(f"  F1@{THRESH}%        : observed {m:.3f} +/- {s:.3f}   vs random {rm:.3f} +/- {rs:.3f}")

    # Paired comparison: observed vs random F1, per protein side
    diffs = [o - r for o, r in zip(obs_f1, rand_f1)]
    n_better = sum(1 for d in diffs if d > 0)
    print(f"\n  BALM-PPI attribution F1@{THRESH}% exceeds the matched random baseline on {n_better}/{len(diffs)} protein sides.")

    try:
        from scipy import stats as scipy_stats
        t_stat, p_val = scipy_stats.ttest_rel(obs_f1, rand_f1)
        w_stat, w_p = scipy_stats.wilcoxon(obs_f1, rand_f1)
        print(f"  Paired t-test (observed vs random F1@{THRESH}%): t={t_stat:.3f}, p={p_val:.2e}")
        print(f"  Wilcoxon signed-rank (observed vs random F1@{THRESH}%): p={w_p:.2e}")
    except ImportError:
        print("  (scipy not available -- install scipy to compute paired significance test)")


if __name__ == "__main__":
    main()
