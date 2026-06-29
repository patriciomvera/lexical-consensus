"""
correlation_diagnostic.py — Step 4 of Phase 1 (checkpoint).

Computes Pearson and Spearman correlation between the upper triangles of the
perceptual and semantic distance matrices (4,950 off-diagonal pairs), generates
a scatter plot, and flags high collinearity (r > 0.85).

This is a DIAGNOSTIC checkpoint (§9.4). After reviewing the report, decide
whether to proceed to Phase 2. The experiment continues regardless of the
collinearity level, but Rule B/C in the pre-registration anticipates the high-r
case honestly.

Outputs:
    <run_dir>/phase1/phase1_correlation_report.json
    <run_dir>/phase1/phase1_scatter.png

Usage:
    python src/phase1/correlation_diagnostic.py [--run-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _shared import CIFAR100_CLASSES, get_run_dir, load_config


def _upper_triangle(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[idx]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="exp_008 Phase 1 — correlation diagnostic"
    )
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    run_dir = get_run_dir(args.run_dir)
    config = load_config()
    flag_threshold = config["phase4"]["collinearity_flag_r"]

    perc_path = run_dir / "phase1" / "perceptual_distance_matrix.csv"
    sem_path  = run_dir / "phase1" / "semantic_distance_matrix.csv"

    if not perc_path.exists():
        raise FileNotFoundError(f"Missing: {perc_path}. Run compute_perceptual_distances.py first.")
    if not sem_path.exists():
        raise FileNotFoundError(f"Missing: {sem_path}. Run compute_semantic_distances.py first.")

    d_perc_df = pd.read_csv(perc_path, index_col=0)
    d_sem_df  = pd.read_csv(sem_path,  index_col=0)

    d_perc_mat = d_perc_df.values
    d_sem_mat  = d_sem_df.values

    d_perc_vals = _upper_triangle(d_perc_mat)
    d_sem_vals  = _upper_triangle(d_sem_mat)

    from scipy.stats import pearsonr, spearmanr
    r_pearson,  p_pearson  = pearsonr(d_perc_vals, d_sem_vals)
    r_spearman, p_spearman = spearmanr(d_perc_vals, d_sem_vals)

    collinearity_flag = bool(abs(r_pearson) > flag_threshold)
    interpretation = (
        f"HIGH COLLINEARITY (|r|={abs(r_pearson):.3f} > {flag_threshold}). "
        "Dissociation power may be low. Rule B likely. Continue but report honestly."
        if collinearity_flag else
        f"Moderate to low collinearity (|r|={abs(r_pearson):.3f}). "
        "Adequate dissociation power expected."
    )

    report = {
        "n_pairs": int(len(d_perc_vals)),
        "pearson_r": round(float(r_pearson), 4),
        "pearson_p": float(p_pearson),
        "spearman_rho": round(float(r_spearman), 4),
        "spearman_p": float(p_spearman),
        "collinearity_flag": collinearity_flag,
        "collinearity_threshold": flag_threshold,
        "interpretation": interpretation,
    }

    # Top 5 most perceptually distant pairs
    idx_perc = np.argsort(_upper_triangle(d_perc_mat))[::-1][:5]
    uti = np.triu_indices(100, k=1)
    top_perc = []
    for k in idx_perc:
        i, j = uti[0][k], uti[1][k]
        top_perc.append({
            "class_u": CIFAR100_CLASSES[i], "class_v": CIFAR100_CLASSES[j],
            "d_perc": round(float(d_perc_mat[i, j]), 4),
            "d_sem":  round(float(d_sem_mat[i, j]),  4),
        })

    # Top 5 most semantically distant pairs
    idx_sem = np.argsort(_upper_triangle(d_sem_mat))[::-1][:5]
    top_sem = []
    for k in idx_sem:
        i, j = uti[0][k], uti[1][k]
        top_sem.append({
            "class_u": CIFAR100_CLASSES[i], "class_v": CIFAR100_CLASSES[j],
            "d_perc": round(float(d_perc_mat[i, j]), 4),
            "d_sem":  round(float(d_sem_mat[i, j]),  4),
        })

    # Top 5 most dissociated pairs (largest |rank_perc - rank_sem|)
    n_pairs = len(d_perc_vals)
    rank_perc = np.argsort(np.argsort(d_perc_vals))
    rank_sem  = np.argsort(np.argsort(d_sem_vals))
    rank_diff = np.abs(rank_perc.astype(int) - rank_sem.astype(int))
    idx_diss = np.argsort(rank_diff)[::-1][:5]
    top_diss = []
    for k in idx_diss:
        i, j = uti[0][k], uti[1][k]
        top_diss.append({
            "class_u": CIFAR100_CLASSES[i], "class_v": CIFAR100_CLASSES[j],
            "d_perc": round(float(d_perc_mat[i, j]), 4),
            "d_sem":  round(float(d_sem_mat[i, j]),  4),
            "rank_gap": int(rank_diff[k]),
        })

    report["top5_most_perceptually_distant"] = top_perc
    report["top5_most_semantically_distant"] = top_sem
    report["top5_most_dissociated"]          = top_diss

    report_path = run_dir / "phase1" / "phase1_correlation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[diagnostic] saved report -> {report_path}")

    # Scatter plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Color by same-superclass (blue) vs different-superclass (red)
    from _shared import CLASS_TO_SUPERCLASS
    same_sc = np.zeros(n_pairs, dtype=bool)
    for k in range(n_pairs):
        i, j = uti[0][k], uti[1][k]
        same_sc[k] = (CLASS_TO_SUPERCLASS.get(CIFAR100_CLASSES[i]) ==
                      CLASS_TO_SUPERCLASS.get(CIFAR100_CLASSES[j]))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(d_perc_vals[~same_sc], d_sem_vals[~same_sc],
               alpha=0.15, s=8, color="red",   label="different superclass")
    ax.scatter(d_perc_vals[same_sc],  d_sem_vals[same_sc],
               alpha=0.40, s=12, color="blue", label="same superclass")
    ax.set_xlabel("d_perc (1 − cosine similarity)", fontsize=11)
    ax.set_ylabel("d_sem  (1 − Wu-Palmer)",          fontsize=11)
    ax.set_title(
        f"Perceptual vs semantic distance (n={n_pairs} pairs)\n"
        f"Pearson r={r_pearson:.3f}, Spearman ρ={r_spearman:.3f}",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()

    scatter_path = run_dir / "phase1" / "phase1_scatter.png"
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print(f"[diagnostic] scatter plot -> {scatter_path}")

    # Console summary
    print(f"\n{'='*55}")
    print(f"  Pearson  r    = {r_pearson:.4f}  (p={p_pearson:.2e})")
    print(f"  Spearman ρ    = {r_spearman:.4f}  (p={p_spearman:.2e})")
    if collinearity_flag:
        print(f"\n  *** WARNING: high collinearity (r > {flag_threshold}) ***")
        print(f"  {interpretation}")
    else:
        print(f"\n  {interpretation}")
    print(f"{'='*55}")
    print(f"\nProceed to Phase 2:\n  python src/phase2/classify_quadrants.py")


if __name__ == "__main__":
    main()
