"""
compute_perceptual_distances.py — Step 2 of Phase 1.

Computes the 100×100 pairwise perceptual distance matrix between CIFAR-100
class centroids in DINOv2-small space:

    centroid_y = normalize( mean{ f_θ(x_i) : x_i ∈ class y } )
    d_perc(u, v) = 1 - cos_sim(centroid_u, centroid_v)

Uses ALL 500 training images per class for centroid stability (not the sampled
episode pool of 200). L2-normalization is applied to each individual embedding
before averaging, then the mean is re-normalized — this matches the exp_007
convention for centroid computation.

Output written to <run_dir>/phase1/perceptual_distance_matrix.csv (100 rows ×
100 columns, index = class names).

Usage:
    python src/phase1/compute_perceptual_distances.py [--run-dir PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _shared import (
    CIFAR100_CLASSES, get_run_dir, load_embeddings, RESULTS_BASE,
)


def compute_class_centroids(
    emb: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> np.ndarray:
    """Compute L2-normalized class centroids. Returns (100, 384) array."""
    dim = emb.shape[1]
    centroids = np.zeros((len(class_names), dim), dtype=np.float64)
    for idx, cls in enumerate(class_names):
        mask = labels == idx
        if not mask.any():
            raise RuntimeError(f"No embeddings found for class '{cls}' (idx={idx})")
        vecs = emb[mask].astype(np.float64)   # (n_cls, 384)
        mean = vecs.mean(axis=0)
        norm = np.linalg.norm(mean)
        centroids[idx] = mean / norm if norm > 0 else mean
    return centroids


def compute_perceptual_distance_matrix(centroids: np.ndarray) -> np.ndarray:
    """100×100 cosine distance matrix from pre-normalized centroids."""
    # Gram matrix: dot products (all centroids are unit-norm)
    gram = centroids @ centroids.T
    gram = np.clip(gram, -1.0, 1.0)
    return 1.0 - gram


def main() -> None:
    parser = argparse.ArgumentParser(
        description="exp_008 Phase 1 — compute perceptual distance matrix"
    )
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    run_dir = get_run_dir(args.run_dir)
    out_path = run_dir / "phase1" / "perceptual_distance_matrix.csv"

    print("[perceptual] loading embeddings ...")
    emb, labels, class_names = load_embeddings()

    print("[perceptual] computing class centroids (all train images) ...")
    centroids = compute_class_centroids(emb, labels, class_names)
    print(f"  centroids shape: {centroids.shape}")

    print("[perceptual] computing 100×100 distance matrix ...")
    dist_matrix = compute_perceptual_distance_matrix(centroids)
    np.fill_diagonal(dist_matrix, 0.0)

    # Sanity check
    assert dist_matrix.shape == (100, 100)
    assert np.allclose(dist_matrix, dist_matrix.T, atol=1e-6), "Matrix not symmetric"
    print(f"  d_perc range: [{dist_matrix.min():.4f}, {dist_matrix.max():.4f}]")
    print(f"  d_perc mean (off-diagonal): {dist_matrix[dist_matrix > 0].mean():.4f}")

    df = pd.DataFrame(dist_matrix, index=class_names, columns=class_names)
    df.to_csv(out_path)
    print(f"[perceptual] saved -> {out_path}")

    # Also save the raw centroids for reuse
    raw_path = run_dir / "phase1" / "raw" / "centroids.npy"
    np.save(raw_path, centroids)
    print(f"[perceptual] centroids saved -> {raw_path}")

    print("\nNext step:\n  python src/phase1/compute_semantic_distances.py")


if __name__ == "__main__":
    main()
