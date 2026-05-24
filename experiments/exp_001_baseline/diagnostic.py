"""
exp_001_baseline/diagnostic.py
-------------------------------
Step 1 of Phase 1: Diagnostic before running the full experiment.

Question: Do CIFAR-10 categories form separable clusters in DINOv2
embedding space?

If yes, the experiment has a chance to work — the perceptual substrate
distinguishes the categories.

If no, the experiment cannot work and we need to either change the
dataset or use a different visual encoder.

Run this BEFORE attempting the full multi-agent experiment.

Usage:
    python -m experiments.exp_001_baseline.diagnostic

Output:
    - Console: separability metrics per CIFAR-10 category pair
    - File:    results/diagnostic_dinov2_cifar10.json
    - Plot:    results/diagnostic_dinov2_cifar10.png (UMAP projection)
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
# 6. Save metrics to results/diagnostic_dinov2_cifar10.json
# 7. Save plot to results/diagnostic_dinov2_cifar10.png
#
# Success criterion: silhouette score > 0.3
# If score is lower, log which category pairs are most confused.

if __name__ == "__main__":
    raise NotImplementedError(
        "Diagnostic script not yet implemented. "
        "This is the first concrete task to build with Claude Code. "
        "See the TODO in this file for the required implementation."
    )
