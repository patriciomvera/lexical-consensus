"""
exp_000_embedding_separability/diagnostic.py
--------------------------------------------
Experiment 0: Embedding Separability Check.

This is a complete, standalone experiment — not a preliminary step.

Question: Do CIFAR-10 categories form separable clusters in DINOv2
embedding space?

This question must be answered before any subsequent experiment is
attempted. If CIFAR-10 categories are not separable in the DINOv2
384-dim space, the centroid-based lexicon (Layer 2) has no geometric
substrate to work with — label acquisition cannot happen.

Outcomes:
    PASS  silhouette score > 0.3 across selected categories
          -> exp_001_single_agent_lexicon may proceed
    FAIL  silhouette score <= 0.3
          -> investigate alternative datasets or encoders before
             building any agent infrastructure

Artifacts written to results/exp_000_embedding_separability/:
    config.yaml  — experiment parameters (categories, n_images, seed)
    metrics.json — per-pair distances, silhouette score, pass/fail flag
    report.md    — human-readable summary with interpretation
    *.png        — UMAP 2D projection for visual inspection

Usage:
    python -m experiments.exp_000_embedding_separability.diagnostic
"""

# TODO — Implementation pending. This script should:
#
# 1. Load 50 images per CIFAR-10 category (no labels exposed to model)
# 2. Encode all images with PerceptionLayer (DINOv2-small)
# 3. For each pair of categories:
#       - Compute mean cosine distance within category (intra-class)
#       - Compute mean cosine distance between categories (inter-class)
#       - Ratio = inter / intra. Higher = more separable.
# 4. Compute silhouette score across all categories
# 5. Generate UMAP 2D projection for visual inspection
# 6. Save all artifacts to results/exp_000_embedding_separability/
#
# Success criterion: silhouette score > 0.3
# If score is lower, log which category pairs are most confused.

if __name__ == "__main__":
    raise NotImplementedError(
        "Experiment 0 not yet implemented. "
        "This is the first concrete task — see TODO above for the "
        "required implementation and PASS/FAIL criteria."
    )
