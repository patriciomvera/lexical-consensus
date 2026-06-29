"""
src.eval
--------
Concept-acquisition evaluation harness for exp_007 (concept carving).

This package holds the reusable, learner-agnostic machinery that turns "a frozen
embedding space + a set of concept definitions" into the statistics the paper
needs. It is deliberately separate from src.metrics (which is exp_004's Shannon
machinery) and from the learners themselves.

Modules
-------
concepts      — Concept / ConceptType definitions and builders
                (NATIVE / DISJUNCTIVE / CROSS_CUTTING / SYNTHETIC_NOVEL).
descriptors   — concept-level "carving" knobs computed in the frozen space:
                NS_silhouette, compactness, centroid_lands_outside, NMI_native.
episodes      — seeded N-way K-shot episode generator (support/query splits).
metrics       — C1, C2, margin, OOV-AUROC, prototype_failure_gap,
                NMI(prediction, object_category), chance.
stats         — bootstrap CI, paired bootstrap, Wilcoxon, effect size.
falsification — random-label, permuted-binding, OOV controls (reuses exp_002
                logic, applied to disjunctive / cross-cutting sets).

Nothing here imports torch or re-encodes images — everything operates on
precomputed 384-d DINOv2 vectors (plan §10).
"""

from __future__ import annotations

__all__ = [
    "concepts",
    "descriptors",
    "episodes",
    "metrics",
    "stats",
    "falsification",
]
