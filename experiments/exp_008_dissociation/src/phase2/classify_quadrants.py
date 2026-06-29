"""
classify_quadrants.py — Step 5 of Phase 2.

Classifies all 4,950 class pairs into four quadrants based on the median of
each distance measure:

    Q1 (concordant-near):   d_perc < med_perc  AND  d_sem < med_sem
    Q2 (concordant-far):    d_perc >= med_perc AND  d_sem >= med_sem
    Q3 (dissociation-A):    d_perc >= med_perc AND  d_sem < med_sem
    Q4 (dissociation-B):    d_perc < med_perc  AND  d_sem >= med_sem

Q3 and Q4 are the dissociation quadrants where the two measures disagree.

Outputs:
    <run_dir>/phase2/raw/all_pairs_quadrants.csv   — all 4,950 pairs + quadrant
    <run_dir>/phase2/quadrant_counts.json          — count per quadrant

Usage:
    python src/phase2/classify_quadrants.py [--run-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _shared import CIFAR100_CLASSES, CLASS_TO_SUPERCLASS, get_run_dir


def classify_pair(d_perc: float, d_sem: float,
                  med_perc: float, med_sem: float) -> str:
    above_perc = d_perc >= med_perc
    above_sem  = d_sem  >= med_sem
    if not above_perc and not above_sem:
        return "Q1"   # concordant-near
    if above_perc and above_sem:
        return "Q2"   # concordant-far
    if above_perc and not above_sem:
        return "Q3"   # dissociation-A (percept-far, sem-near)
    # not above_perc and above_sem
    return "Q4"       # dissociation-B (percept-near, sem-far)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="exp_008 Phase 2 — classify quadrants"
    )
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    run_dir = get_run_dir(args.run_dir)

    perc_path = run_dir / "phase1" / "perceptual_distance_matrix.csv"
    sem_path  = run_dir / "phase1" / "semantic_distance_matrix.csv"

    d_perc_df = pd.read_csv(perc_path, index_col=0)
    d_sem_df  = pd.read_csv(sem_path,  index_col=0)

    d_perc_mat = d_perc_df.values.astype(np.float64)
    d_sem_mat  = d_sem_df.values.astype(np.float64)

    # Upper triangle indices for all 4,950 pairs
    uti = np.triu_indices(100, k=1)
    d_perc_vals = d_perc_mat[uti]
    d_sem_vals  = d_sem_mat[uti]

    med_perc = float(np.median(d_perc_vals))
    med_sem  = float(np.median(d_sem_vals))
    print(f"[quadrants] median d_perc = {med_perc:.4f}")
    print(f"[quadrants] median d_sem  = {med_sem:.4f}")

    rows = []
    for k in range(len(d_perc_vals)):
        i, j = int(uti[0][k]), int(uti[1][k])
        cu, cv = CIFAR100_CLASSES[i], CIFAR100_CLASSES[j]
        dp, ds = float(d_perc_vals[k]), float(d_sem_vals[k])
        q = classify_pair(dp, ds, med_perc, med_sem)
        rows.append({
            "pair_id":         k,
            "class_u":         cu,
            "class_v":         cv,
            "superclass_u":    CLASS_TO_SUPERCLASS.get(cu, "unknown"),
            "superclass_v":    CLASS_TO_SUPERCLASS.get(cv, "unknown"),
            "same_superclass": CLASS_TO_SUPERCLASS.get(cu) == CLASS_TO_SUPERCLASS.get(cv),
            "d_perc":          round(dp, 6),
            "d_sem":           round(ds, 6),
            "quadrant":        q,
        })

    df = pd.DataFrame(rows)

    raw_path = run_dir / "phase2" / "raw" / "all_pairs_quadrants.csv"
    df.to_csv(raw_path, index=False)
    print(f"[quadrants] full table ({len(df)} pairs) -> {raw_path}")

    counts = df["quadrant"].value_counts().to_dict()
    counts_ordered = {q: counts.get(q, 0) for q in ["Q1", "Q2", "Q3", "Q4"]}
    counts_meta = {
        "total_pairs": len(df),
        "median_d_perc": round(med_perc, 6),
        "median_d_sem":  round(med_sem,  6),
        "Q1_concordant_near":   counts_ordered["Q1"],
        "Q2_concordant_far":    counts_ordered["Q2"],
        "Q3_dissociation_A":    counts_ordered["Q3"],
        "Q4_dissociation_B":    counts_ordered["Q4"],
    }

    counts_path = run_dir / "phase2" / "quadrant_counts.json"
    with open(counts_path, "w") as f:
        json.dump(counts_meta, f, indent=2)
    print(f"[quadrants] counts -> {counts_path}")

    for q in ["Q1", "Q2", "Q3", "Q4"]:
        label = {
            "Q1": "concordant-near (both close)",
            "Q2": "concordant-far  (both far)  ",
            "Q3": "dissoc-A        (perc-far, sem-near)",
            "Q4": "dissoc-B        (perc-near, sem-far)",
        }[q]
        print(f"  {q} {label}: {counts_ordered[q]:>5}")

    print(f"\nNext step:\n  python src/phase2/sample_pairs.py")


if __name__ == "__main__":
    main()
