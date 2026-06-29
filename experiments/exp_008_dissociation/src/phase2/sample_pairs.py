"""
sample_pairs.py — Step 6 of Phase 2.

Stratified random sampling: 25 pairs from each of the 4 quadrants (100 total).
If a quadrant has fewer than 25 pairs, takes all available and logs the count.

Sampling is seeded at 42 (config: phase2.sample_seed) for reproducibility.

Also runs the backward-compatibility check: for each CIFAR-10 class pair that
was used as a near/mid/far tier in exp_007b, records its d_perc and d_sem and
notes which quadrant it falls in (sanity check, not a new analysis).

Outputs:
    <run_dir>/phase2/sampled_pairs.csv       — 100 pairs with quadrant + distances

Usage:
    python src/phase2/sample_pairs.py [--run-dir PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _shared import get_run_dir, load_config


# CIFAR-10 categories that exp_007b used as tier examples — sanity check only.
# These 10 classes are a subset of CIFAR-100; confirm they appear in the table.
CIFAR10_CLASSES = {
    "frog", "horse", "ship", "automobile", "airplane",
    "bird", "cat", "deer", "dog", "truck",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="exp_008 Phase 2 — sample pairs from each quadrant"
    )
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    run_dir = get_run_dir(args.run_dir)
    config = load_config()
    n_per_quadrant = config["phase2"]["n_pairs_per_quadrant"]
    seed           = config["phase2"]["sample_seed"]
    min_warn       = config["phase2"]["min_pairs_per_quadrant"]

    all_path = run_dir / "phase2" / "raw" / "all_pairs_quadrants.csv"
    if not all_path.exists():
        raise FileNotFoundError(f"Missing: {all_path}. Run classify_quadrants.py first.")

    df = pd.read_csv(all_path)
    rng = np.random.default_rng(seed)

    sampled_parts = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        subset = df[df["quadrant"] == q].reset_index(drop=True)
        available = len(subset)
        take = min(n_per_quadrant, available)
        if available < min_warn:
            print(f"  *** WARNING: quadrant {q} has only {available} pairs "
                  f"(< min_warn={min_warn}). Taking all {available}.")
        elif available < n_per_quadrant:
            print(f"  NOTE: {q} has {available} pairs; requested {n_per_quadrant} — "
                  f"taking all {available}.")
        chosen_idx = rng.choice(available, size=take, replace=False)
        chosen_idx.sort()
        sampled_parts.append(subset.iloc[chosen_idx])
        print(f"  {q}: sampled {take}/{available} pairs")

    sampled = pd.concat(sampled_parts, ignore_index=True)
    # Re-index pair_id to be sequential within the sampled set for downstream use
    sampled = sampled.reset_index(drop=True)
    sampled.insert(0, "sample_id", sampled.index)

    out_path = run_dir / "phase2" / "sampled_pairs.csv"
    sampled.to_csv(out_path, index=False)
    print(f"[sample] saved {len(sampled)} pairs -> {out_path}")

    # Backward-compatibility check
    cifar10_in_sample = sampled[
        sampled["class_u"].isin(CIFAR10_CLASSES) |
        sampled["class_v"].isin(CIFAR10_CLASSES)
    ]
    print(f"\n[compat] {len(cifar10_in_sample)} sampled pairs involve CIFAR-10 classes:")
    if len(cifar10_in_sample) > 0:
        print(cifar10_in_sample[["sample_id", "class_u", "class_v",
                                  "d_perc", "d_sem", "quadrant"]].to_string(index=False))

    print(f"\nNext step:\n  python src/phase3/run_episodes.py")


if __name__ == "__main__":
    main()
